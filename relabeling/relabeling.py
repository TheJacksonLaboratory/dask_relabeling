import os
import shutil
import pathlib
import zipfile

import operator
import functools

import numpy as np

import dask.array as da
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from numcodecs import JSON, Blosc

from typing import List, Union, Callable

from . import utils
from . import chunkops


def _segment_overlapping(img: da.Array, seg_fn: Callable, overlaps: List[int],
                         ndim: int = 2,
                         segmentation_fn_kwargs: Union[dict, None] = None,
                         persist: Union[bool, pathlib.Path] = False,
                         progressbar: bool = False
                         ) -> da.Array:

    if segmentation_fn_kwargs is None:
        segmentation_fn_kwargs = {}

    img_overlapped = da.overlap.overlap(
        img,
        depth=tuple([(0, 0)] * (img.ndim - ndim)
                    + [(overlap, overlap) for overlap in overlaps]),
        boundary=None,
    )

    labeled_chunks = tuple(
        [tuple(cs + (overlap if location > 0 else 0)
               + (overlap if location < len(cs_axis) - 1 else 0)
               for location, cs in enumerate(cs_axis))
         for overlap, cs_axis in zip(overlaps, img.chunks[-ndim:])
         ]
    )

    block_labeled = da.map_blocks(
        chunkops.segmentation_wrapper,
        img_overlapped,
        seg_fn,
        **segmentation_fn_kwargs,
        chunks=labeled_chunks,
        drop_axis=tuple(range(img.ndim - ndim)),
        dtype=np.int32,
        meta=np.empty((0, 0), dtype=np.int32)
    )

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            padding = utils.save_intermediate_array(
                block_labeled,
                filename="temp_labeled.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            block_labeled = utils.load_intermediate_array(
                filename=persist / "temp_labeled.zarr",
                padding=padding
            )

        elif persist:
            block_labeled = block_labeled.persist()

    return block_labeled


def _remove_overlapped_labels(labels: da.Array, overlaps: List[int],
                              threshold: float = 0.05,
                              ndim: int = 2,
                              relabel: bool = False,
                              persist: Union[bool, pathlib.Path] = False,
                              progressbar: bool = False) -> da.Array:

    block_relabeled = da.map_blocks(
        chunkops.remove_overlapped_objects,
        labels,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        dtype=np.int32,
        meta=np.empty((0, ), dtype=np.int32)
    )

    if relabel:
        block_relabeled = da.map_blocks(
            chunkops.sort_indices,
            block_relabeled,
            ndim=ndim,
            dtype=np.int32,
            meta=np.empty((0, ), dtype=np.int32)
        )

        # Relabel each chunk to have globally unique label indices.
        total = 0

        for relabeled_block in map(
          functools.partial(operator.getitem, block_relabeled),
          da.core.slices_from_chunks(block_relabeled.chunks)):
            n = da.max(relabeled_block)

            block_label_offset = da.where(relabeled_block > 0, total, 0)
            relabeled_block += block_label_offset

            total += n

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            padding = utils.save_intermediate_array(
                block_relabeled,
                filename="temp_relabeled.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            block_relabeled = utils.load_intermediate_array(
                filename=persist / "temp_relabeled.zarr",
                padding=padding
            )

            shutil.rmtree(persist / "temp_labeled.zarr")

        elif persist:
            block_relabeled = block_relabeled.persist()

    return block_relabeled


def _merge_overlapped_tiles(labels: da.Array, overlaps: List[int],
                            ndim: int = 2,
                            to_geojson: bool = False,
                            persist: Union[bool, pathlib.Path] = False,
                            progressbar: bool = False) -> da.Array:

    if to_geojson:
        merged_chunks = labels.chunks
    else:
        merged_chunks = tuple(
            list(labels.chunks[:(labels.ndim - ndim)])
            + [tuple(cs
                     - (overlap if location > 0 else 0)
                     - (overlap if location < len(cs_axis) - 1 else 0)
                     for location, cs in enumerate(cs_axis))
               for overlap, cs_axis in zip(overlaps, labels.chunks[-ndim:])
               ]
        )

    merged_depth = tuple([0] * (labels.ndim - ndim)
                         + [(overlap, overlap) for overlap in overlaps])

    # Merge the overlapped objects from adjacent chunks for all chunk tiles.
    merged_tiles = da.map_overlap(
        chunkops.merge_tiles,
        labels,
        overlaps=overlaps,
        to_geojson=to_geojson,
        depth=merged_depth,
        boundary=None,
        trim=False,
        chunks=merged_chunks,
        dtype=np.int32,
        meta=np.empty((0, 0), dtype=np.int32)
    )

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            padding = utils.save_intermediate_array(
                merged_tiles,
                filename="temp_merged.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            merged_tiles = utils.load_intermediate_array(
                filename=persist / "temp_merged.zarr",
                padding=padding
            )

            shutil.rmtree(persist / "temp_relabeled.zarr")

        elif persist:
            merged_tiles = merged_tiles.persist()

    return merged_tiles


def _merge_features_tiles(labels: da.Array,
                          overlaps: List[int],
                          object_classes: Union[dict, None] = None,
                          ndim: int = 2,
                          out_dir: Union[pathlib.Path, None] = None,
                          persist: Union[bool, pathlib.Path] = False,
                          progressbar: bool = False
                          ) -> Union[da.Array, pathlib.Path]:
    if object_classes is None:
        object_classes = {
            0: "cell"
        }

    labels_features = da.map_blocks(
        chunkops.compute_features,
        labels,
        overlaps=overlaps,
        object_classes=object_classes,
        ndim=ndim,
        chunks=(1, 1),
        dtype=object,
        meta=np.empty((0, 0), dtype=object)
    )

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)

        with progressbar_callback():
            if isinstance(persist, pathlib.Path):
                padding = utils.save_intermediate_array(
                    labels_features,
                    filename="temp_features.zarr",
                    out_dir=persist,
                    object_codec=JSON()
                )

                labels_features = utils.load_intermediate_array(
                    filename=persist / "temp_features.zarr",
                    padding=padding
                )

                shutil.rmtree(persist / "temp_merged.zarr")

            elif persist:
                labels_features = labels_features.persist()

        filenames = da.map_blocks(
            chunkops.dump_annotation,
            labels_features,
            out_dir=out_dir,
            dtype=object,
            meta=np.empty((0, ), dtype=object)
        )

        filenames.compute()

        if isinstance(persist, pathlib.Path):
            shutil.rmtree(persist / "temp_features.zarr")

        out_fn = pathlib.Path(str(out_dir) + ".zip")
        with zipfile.ZipFile(out_fn, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for file_path in out_dir.rglob("*.geojson"):
                archive.write(file_path,
                              arcname=file_path.relative_to(out_dir))

        shutil.rmtree(out_dir)

        labels_features = out_fn

    return labels_features


def _prepare_input(img: da.Array, ndim: int = 2) -> da.Array:
    # Prepare input for overlap.
    padding = [(0, 0)] * (img.ndim - ndim)
    padding += [(0, (cs - dim) % cs)
                for dim, cs in zip(img.shape[-ndim:], img.chunksize[-ndim:])]

    if any(map(any, padding)):
        img_padded = da.pad(
            img,
            padding
        )

        img_rechunked = da.rechunk(img_padded, img.chunksize)

    else:
        img_rechunked = img

    return img_rechunked


def label(img: da.Array, seg_fn: Callable,
          overlaps: Union[int, List[int]] = 50,
          threshold: float = 0.05,
          ndim: int = 2,
          sort_labels: bool = False,
          persist: Union[bool, pathlib.Path, str] = False,
          progressbar: bool = False,
          segmentation_fn_kwargs: Union[dict, None] = None
          ) -> da.Array:

    if isinstance(overlaps, int):
        overlaps = [overlaps] * ndim

    if isinstance(persist, str):
        persist = pathlib.Path(persist)

    img_rechunked = _prepare_input(img, ndim=ndim)

    labels = _segment_overlapping(
        img_rechunked,
        seg_fn=seg_fn,
        overlaps=overlaps,
        ndim=ndim,
        segmentation_fn_kwargs=segmentation_fn_kwargs,
        persist=persist,
        progressbar=progressbar
    )

    labels = _remove_overlapped_labels(
        labels,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        relabel=not sort_labels,
        persist=persist,
        progressbar=progressbar
    )

    labels = _merge_overlapped_tiles(
        labels,
        overlaps=overlaps,
        ndim=ndim,
        to_geojson=sort_labels,
        persist=persist,
        progressbar=progressbar
    )

    return labels


def dump_to_geojson(labels: da.Array, overlaps: Union[int, List[int]] = 50,
                    ndim: int = 2,
                    object_classes: Union[dict, None] = None,
                    out_dir: Union[pathlib.Path, str, None] = None,
                    persist: Union[bool, pathlib.Path, str] = False,
                    progressbar: bool = False,
                    ) -> None:
    if isinstance(overlaps, int):
        overlaps = [overlaps] * ndim

    if isinstance(persist, str):
        persist = pathlib.Path(persist)

    if out_dir is not None and not isinstance(out_dir, pathlib.Path):
        out_dir = pathlib.Path(out_dir)

    if object_classes is None:
        classes_ids = range(labels.shape[:(labels.ndim - ndim)]
                            if labels.ndim - ndim > 0 else 1)
        object_classes = {class_id: "cell" for class_id in classes_ids}

    labels = _merge_features_tiles(
        labels,
        overlaps=overlaps,
        object_classes=object_classes,
        ndim=ndim,
        out_dir=out_dir,
        persist=persist,
        progressbar=progressbar
    )

    return labels
