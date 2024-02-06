import os
import shutil
import pathlib
import zipfile

import numpy as np
import dask.array as da
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from numcodecs import JSON, Blosc

from typing import List, Union, Callable

from . import utils
from . import chunkops

# TODO: Remove progressbar utility

def segment_overlapped_input(img: da.Array, seg_fn: Callable,
                             ndim: int = 2,
                             returns_classes: bool = False,
                             segmentation_fn_kwargs: Union[dict, None] = None,
                             persist: Union[bool, pathlib.Path] = False,
                             progressbar: bool = False) -> da.Array:

    if segmentation_fn_kwargs is None:
        segmentation_fn_kwargs = {}

    if returns_classes:
        labeled_chunks = [(2, )]
    else:
        labeled_chunks = []

    labeled_chunks += list(img.chunks)[-ndim:]

    labeled = da.map_blocks(
        seg_fn,
        img,
        **segmentation_fn_kwargs,
        chunks=tuple(labeled_chunks),
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
                labeled,
                filename="temp_labeled.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            labeled = utils.load_intermediate_array(
                filename=persist / "temp_labeled.zarr",
                padding=padding
            )

        elif persist:
            labeled = labeled.persist()

    return labeled


def remove_overlapped_labels(labels: da.Array, overlaps: List[int],
                             threshold: float = 0.05,
                             ndim: int = 2,
                             persist: Union[bool, pathlib.Path] = False,
                             progressbar: bool = False) -> da.Array:

    removed = da.map_blocks(
        chunkops.remove_overlapped_objects,
        labels,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        dtype=np.int32,
        meta=np.empty((0, ), dtype=np.int32)
    )

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            padding = utils.save_intermediate_array(
                removed,
                filename="temp_removed.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            removed = utils.load_intermediate_array(
                filename=persist / "temp_removed.zarr",
                padding=padding
            )

            if os.path.isdir(persist / "temp_labeled.zarr"):
                shutil.rmtree(persist / "temp_labeled.zarr")

        elif persist:
            removed = removed.persist()

    return removed


def sort_overlapped_labels(labels: da.Array, ndim: int = 2,
                           persist: Union[bool, pathlib.Path] = False,
                           progressbar: bool = False) -> da.Array:
    sorted = da.map_blocks(
        chunkops.sort_indices,
        labels,
        ndim=ndim,
        dtype=np.int32,
        meta=np.empty((0, ), dtype=np.int32)
    )

    # Relabel each chunk to have globally unique label indices.
    total = 0

    sorted_globally = da.zeros_like(sorted)

    indices = da.core.slices_from_chunks(sorted.chunks)

    for idx in indices:
        n = da.max(sorted[idx])

        label_offset = da.where(sorted[idx] > 0, total, 0)
        sorted_globally[idx] = sorted[idx] + label_offset

        total += n

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            padding = utils.save_intermediate_array(
                sorted_globally,
                filename="temp_sorted.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            sorted_globally = utils.load_intermediate_array(
                filename=persist / "temp_sorted.zarr",
                padding=padding
            )

            if os.path.isdir(persist / "temp_removed.zarr"):
                shutil.rmtree(persist / "temp_removed.zarr")

        elif persist:
            sorted_globally = sorted_globally.persist()

    return sorted_globally


def merge_overlapped_tiles(labels: da.Array, overlaps: List[int],
                           ndim: int = 2,
                           persist: Union[bool, pathlib.Path] = False,
                           progressbar: bool = False) -> da.Array:

    merged_chunks = tuple(
        list(labels.chunks[:(labels.ndim - ndim)])
        + [tuple(cs
                 - (overlap if location > 0 else 0)
                 - (overlap if location < len(cs_axis) - 1 else 0)
                 for location, cs in enumerate(cs_axis)
                 )
           for overlap, cs_axis in zip(overlaps, labels.chunks[-ndim:])
           ]
    )

    merged_depth = tuple([0] * (labels.ndim - ndim)
                         + [(overlap, overlap) for overlap in overlaps])

    # Merge the overlapped objects from adjacent chunks for all chunk tiles.
    merged = da.map_overlap(
        chunkops.merge_tiles,
        labels,
        overlaps=overlaps,
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
                merged,
                filename="temp_merged.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

            merged = utils.load_intermediate_array(
                filename=persist / "temp_merged.zarr",
                padding=padding
            )

            if os.path.isdir(persist / "temp_sorted.zarr"):
                shutil.rmtree(persist / "temp_sorted.zarr")

        elif persist:
            merged = merged.persist()

    return merged


def annotate_labeled_tiles(labels: da.Array, overlaps: List[int],
                           object_classes: Union[dict, None] = None,
                           ndim: int = 2,
                           persist: Union[bool, pathlib.Path] = False,
                           progressbar: bool = False
                           ) -> Union[da.Array, pathlib.Path]:
    if object_classes is None:
        object_classes = {
            0: "cell"
        }

    labels_annotations = da.map_blocks(
        chunkops.annotate_object_fetures,
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

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            padding = utils.save_intermediate_array(
                labels_annotations,
                filename="temp_annotations.zarr",
                out_dir=persist,
                object_codec=JSON()
            )

            labels_annotations = utils.load_intermediate_array(
                filename=persist / "temp_annotations.zarr",
                padding=padding
            )

            if os.path.isdir(persist / "temp_removed.zarr"):
                shutil.rmtree(persist / "temp_removed.zarr")

        elif persist:
            labels_annotations = labels_annotations.persist()

    return labels_annotations


def zip_annotated_labeled_tiles(labels: da.Array, out_dir: pathlib.Path,
                                persist: Union[bool, pathlib.Path] = False,
                                progressbar: bool = False) -> pathlib.Path:

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    geojson_filenames = da.map_blocks(
        chunkops.dump_annotaions,
        labels,
        out_dir=out_dir,
        chunks=(1, 1),
        dtype=object,
        meta=np.empty((0, 0), dtype=object)
    )

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        geojson_filenames = geojson_filenames.compute()

    out_zip_filename = pathlib.Path(str(out_dir) + ".zip")
    with zipfile.ZipFile(out_zip_filename, "w", zipfile.ZIP_DEFLATED,
                         compresslevel=9) as out_zip:
        for chunk_filename in geojson_filenames.flatten().tolist():
            if chunk_filename:
                out_zip.write(chunk_filename,
                              arcname=chunk_filename.relative_to(out_dir))

    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    if isinstance(persist, pathlib.Path):
        if os.path.isdir(persist / "temp_annotations.zarr"):
            shutil.rmtree(persist / "temp_annotations.zarr")

    return out_zip_filename


def prepare_input(img: da.Array, overlaps: List[int], ndim: int = 2
                  ) -> da.Array:
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

    img_overlapped = da.overlap.overlap(
        img_rechunked,
        depth=tuple([(0, 0)] * (img.ndim - ndim)
                    + [(overlap, overlap) for overlap in overlaps]),
        boundary=None,
    )

    return img_overlapped


def image2labels(img: da.Array, seg_fn: Callable,
                 overlaps: Union[int, List[int]] = 50,
                 threshold: float = 0.05,
                 ndim: int = 2,
                 persist: Union[bool, pathlib.Path, str] = False,
                 progressbar: bool = False,
                 segmentation_fn_kwargs: Union[dict, None] = None) -> da.Array:

    if isinstance(overlaps, int):
        overlaps = [overlaps] * ndim

    if isinstance(persist, str):
        persist = pathlib.Path(persist)

    img_overlapped = prepare_input(img, overlaps=overlaps, ndim=ndim)

    labels = segment_overlapped_input(
        img_overlapped,
        seg_fn=seg_fn,
        ndim=ndim,
        segmentation_fn_kwargs=segmentation_fn_kwargs,
        persist=persist,
        progressbar=progressbar
    )

    labels = remove_overlapped_labels(
        labels,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        persist=persist,
        progressbar=progressbar
    )

    labels = sort_overlapped_labels(
        labels,
        ndim=ndim,
        persist=persist,
        progressbar=progressbar
    )

    labels = merge_overlapped_tiles(
        labels,
        overlaps=overlaps,
        ndim=ndim,
        persist=persist,
        progressbar=progressbar
    )

    return labels


def labels2geojson(labels: da.Array, overlaps: Union[int, List[int]] = 50,
                   threshold: float = 0.05,
                   ndim: int = 2,
                   object_classes: Union[dict, None] = None,
                   out_dir: Union[pathlib.Path, str, None] = None,
                   persist: Union[bool, pathlib.Path, str] = False,
                   progressbar: bool = False,
                   pre_overlapped: bool = False) -> None:

    if isinstance(overlaps, int):
        overlaps = [overlaps] * ndim

    if isinstance(persist, str):
        persist = pathlib.Path(persist)

    if isinstance(out_dir, str):
        out_dir = pathlib.Path(out_dir)

    if not pre_overlapped:
        labels = prepare_input(labels, ndim=ndim)

        labels = da.overlap.overlap(
            labels,
            depth=tuple([(0, 0)] * (labels.ndim - ndim)
                        + [(overlap, overlap) for overlap in overlaps]),
            boundary=None
        )

    labels = remove_overlapped_labels(
        labels,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        persist=persist,
        progressbar=progressbar
    )

    if object_classes is None:
        classes_ids = range(labels.shape[:(labels.ndim - ndim)]
                            if labels.ndim - ndim > 0 else 1)
        object_classes = {class_id: "cell" for class_id in classes_ids}

    labels = annotate_labeled_tiles(
        labels,
        overlaps=overlaps,
        object_classes=object_classes,
        ndim=ndim,
        persist=persist,
        progressbar=progressbar
    )

    if out_dir is not None:
        zip_annotated_labeled_tiles(labels, out_dir, progressbar=progressbar)

    return labels


def image2geojson(img: da.Array, seg_fn: Callable,
                  overlaps: Union[int, List[int]] = 50,
                  threshold: float = 0.05,
                  ndim: int = 2,
                  object_classes: Union[dict, None] = None,
                  out_dir: Union[pathlib.Path, str, None] = None,
                  persist: Union[bool, pathlib.Path, str] = False,
                  progressbar: bool = False,
                  segmentation_fn_kwargs: Union[dict, None] = None) -> None:

    if isinstance(overlaps, int):
        overlaps = [overlaps] * ndim

    if isinstance(persist, str):
        persist = pathlib.Path(persist)

    img_overlapped = prepare_input(img, overlaps=overlaps, ndim=ndim)

    labels = segment_overlapped_input(
        img_overlapped,
        seg_fn=seg_fn,
        ndim=ndim,
        segmentation_fn_kwargs=segmentation_fn_kwargs,
        persist=persist,
        progressbar=progressbar
    )

    labels = labels2geojson(
        labels,
        overlaps,
        threshold=threshold,
        ndim=ndim,
        object_classes=object_classes,
        out_dir=out_dir,
        persist=persist,
        progressbar=progressbar,
        pre_overlapped=True
    )

    return labels
