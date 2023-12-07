import os
import shutil
import pathlib
import zipfile

import operator
import functools
import itertools

import numpy as np
import scipy
from skimage import transform

import dask
import dask.array as da
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
from numcodecs import JSON, Pickle, Blosc

from typing import List, Tuple, Union, Callable

from .utils import save_intermediate_array, dump_annotations


def _segmentation_function(img_chunk: np.ndarray,
                           segmentation_fn: Callable,
                           mask: Union[np.ndarray, None] = None,
                           ndim: int = 2,
                           **segmentation_fn_kwargs) -> np.ndarray:
    """Wrapper to apply a segmentation function to each chunk.
    If a mask is passed, only the segmentation corresponding to the masked
    regions are returned.
    """
    if mask is not None and mask.sum() == 0:
        return np.zeros(img_chunk.shape[-ndim:], dtype=np.int32)

    # Execute the segmentation process
    labeled_image = segmentation_fn(img_chunk, **segmentation_fn_kwargs)
    labeled_image = labeled_image.astype(np.int32)

    if mask is not None:
        labeled_image = np.where(mask, labeled_image, 0)

    return labeled_image


def _remove_overlapped_objects(labeled_image: np.array, overlap: List[int],
                               chunk_location: Tuple[int],
                               num_chunks: Tuple[int],
                               threshold: float = 0.05,
                               ndim: int = 2,
                               ) -> np.ndarray:
    """Removes ambiguous objects detected in overlapping regions between
    adjacent chunks.
    """
    if labeled_image.ndim > ndim:
        classes = labeled_image[1]
        labeled_image = labeled_image[0]
    else:
        classes = None

    # Remove objects touching the `lower` and `upper` borders of the current
    # axis from labels of this chunk.
    overlapped_labels = []

    overlapped_sel = itertools.product(
        [True, False],
        zip(range(ndim), chunk_location, num_chunks, overlap)
    )

    for lvl, (i, loc, nchk, ovp) in overlapped_sel:
        if loc > 0 and lvl:
            in_sel = tuple(
                [slice(None)] * i
                + [slice(ovp, 2 * ovp)]
                + [slice(None)] * (ndim - 1 - i)
            )
            out_sel = tuple(
                [slice(None)] * i
                + [slice(0, ovp)]
                + [slice(None)] * (ndim - 1 - i)
            )

        elif loc < nchk - 1 and not lvl:
            in_sel = tuple(
                [slice(None)] * i
                + [slice(-2 * ovp, -ovp)]
                + [slice(None)] * (ndim - 1 - i)
            )
            out_sel = tuple(
                [slice(None)] * i
                + [slice(-ovp, None)]
                + [slice(None)] * (ndim - 1 - i)
            )
        else:
            continue

        in_margin = labeled_image[in_sel]
        out_margin = labeled_image[out_sel]

        for mrg_l in np.unique(out_margin):
            if mrg_l == 0:
                continue

            in_mrg = np.sum(in_margin == mrg_l)
            out_mrg = np.sum(out_margin == mrg_l)
            out_mrg_prop = out_mrg / (in_mrg + out_mrg)
            in_mrg_prop = 1.0 - out_mrg_prop

            if (loc % 2 != 0 and out_mrg_prop >= threshold
               or loc % 2 == 0 and in_mrg_prop < threshold):
                overlapped_labels.append(mrg_l)

    for mrg_l in overlapped_labels:
        if classes is not None:
            classes = np.where(labeled_image == mrg_l, 0, classes)

        labeled_image = np.where(labeled_image == mrg_l, 0, labeled_image)

    # Relabel image to have a continuous sequence of indices
    labels_map = np.unique(labeled_image)
    labels_map = scipy.sparse.csr_matrix(
        (np.arange(labels_map.size, dtype=np.int32),
            (labels_map, np.zeros_like(labels_map))),
        shape=(labels_map.max() + 1, 1)
    )

    if ndim == 2:
        labeled_image = labeled_image[None, ...]

    relabeled_image = np.empty_like(labeled_image)

    # Sparse matrices cann only be reshaped into 2D matrices, so treat each
    # stack as a different image.
    for l_idx, l_img in enumerate(labeled_image):
        labels_map = labels_map[l_img.ravel()]
        rl_img = labels_map.reshape(l_img.shape)
        rl_img.todense(out=relabeled_image[l_idx, ...])

    if ndim > 2:
        labeled_image = np.stack(relabeled_image, axis=0)
    else:
        labeled_image = relabeled_image[0]

    num_labels = np.max(labeled_image)

    if classes is not None:
        labeled_image = np.stack((labeled_image, classes), axis=0)

    return labeled_image, num_labels


def _delayed_remove_overlapped_objects(labeled_image: da.Array,
                                       overlap: List[int],
                                       chunk_location: Tuple[int],
                                       num_chunks: Tuple[int],
                                       threshold: float = 0.05,
                                       ndim: int = 2,
                                       ) -> da.Array:
    """Delayed version of `remove_overlapped_objects` that allows to run the
    removal process lazily.
    """
    removed_overlaps = dask.delayed(_remove_overlapped_objects, nout=2)

    labeled_block, n = removed_overlaps(labeled_image, overlap=overlap,
                                        chunk_location=chunk_location,
                                        num_chunks=num_chunks,
                                        threshold=threshold,
                                        ndim=ndim)

    n = dask.delayed(np.int32)(n)

    relabeled = da.from_delayed(labeled_block, shape=labeled_image.shape,
                                dtype=np.int32)
    n = da.from_delayed(n, shape=(), dtype=np.int32)
    return relabeled, n


def _merge_tiles(tile: np.ndarray, overlap: List[int],
                 to_geojson: bool = False,
                 block_info: Union[dict, None] = None) -> np.ndarray:
    """Merge objects detected in overlapping regions from adjacent chunks of
    this.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    ndim = tile.ndim

    # Compute selections from regions to merge
    base_src_sel = tuple(
        slice(ovp if loc > 0 else 0, -ovp if loc < nchk - 1 else None)
        for loc, nchk, ovp in zip(chunk_location, num_chunks, overlap)
    )

    merging_tile = np.copy(tile[base_src_sel])

    if to_geojson:
        offset_index = np.max(merging_tile)
        offset_labels = np.where(merging_tile, offset_index + 1, 0)
        merging_tile = merging_tile - offset_labels

    left_tile = np.empty_like(merging_tile)

    valid_indices = []
    for d in range(ndim):
        for comb, k in itertools.product(
          itertools.combinations(range(ndim), d), range(2 ** (ndim - d))):
            indices = list(np.unpackbits(np.array([k], dtype=np.uint8),
                                         count=ndim - d,
                                         bitorder="little"))
            for fixed in comb:
                indices[fixed:fixed] = [None]

            if any(map(lambda lvl, loc, nchk:
                       lvl is not None and (loc % 2 == 0
                                            or loc == 0 and not lvl
                                            or loc >= nchk - 1 and lvl),
                       indices, chunk_location, num_chunks)):
                continue

            valid_indices.append(indices)

    valid_src_sel = (
        map(lambda loc, nchk, ovp, level:
            slice(ovp if loc > 0 else 0, -ovp if loc < nchk - 1 else None)
            if level is None else
            slice(0, ovp) if not level else slice(-ovp, None),
            chunk_location, num_chunks, overlap, indices)
        for indices in valid_indices
    )

    valid_dst_sel = (
        map(lambda loc, nchk, ovp, level:
            slice(None)
            if level is None else
            slice(ovp if loc > 0 else 0, ovp * (2 if loc > 0 else 1))
            if not level else
            slice(-ovp * (2 if loc < nchk - 1 else 1),
                  -ovp if loc < nchk - 1 else None),
            chunk_location, num_chunks, overlap, indices)
        for indices in valid_indices
    )

    for src_sel, dst_sel in zip(valid_src_sel, valid_dst_sel):
        left_tile[:] = 0
        left_tile[tuple(dst_sel)] = tile[tuple(src_sel)]

        for l_label in np.unique(left_tile):
            if l_label == 0:
                continue

            merging_tile = np.where(
                left_tile == l_label,
                0 if to_geojson else l_label,
                merging_tile
            )

    if to_geojson:
        merged_tile = merging_tile
    else:
        merged_tile = merging_tile[base_src_sel]

    return merged_tile


def _dump_chunk_geojson(merged_tile: np.ndarray, overlap: List[int],
                        object_classes: dict,
                        out_dir: Union[pathlib.Path, None] = None,
                        ndim: int = 2,
                        block_info: Union[dict, None] = None) -> np.ndarray:
    """Convert detections into GeoJson Feature objects containing the contour
    and class of each detected object in this chunk.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = [
        s - (ovp if loc > 0 else 0) - (ovp if loc < nchk - 1 else 0)
        for s, loc, nchk, ovp in zip(merged_tile.shape,
                                     chunk_location,
                                     num_chunks,
                                     overlap)
    ]

    offset = np.array(chunk_location, dtype=np.int64)
    offset *= np.array(chunk_shape)
    offset -= np.array([ovp if loc > 0 else 0
                        for loc, ovp in zip(chunk_location, overlap)])
    offset = offset[[1, 0]]

    out_fn = "detections-" + "-".join(map(str, chunk_location)) + ".geojson"
    out_fn = out_dir / out_fn

    # TODO: Pass predicted classes as additional dask.Array, and object
    # types as dictionaries
    detections = dump_annotations(
        merged_tile,
        object_classes=object_classes,
        filename=out_fn,
        scale=1.0,
        offset=offset,
        ndim=ndim,
        keep_all=False
    )

    merged_tile = np.array([[detections]], dtype=object)

    return merged_tile


def _segment_overlapping(img: da.Array, seg_fn: Callable, overlap: List[int],
                         mask: Union[da.Array, None] = None,
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
                    + [(ovp, ovp) for ovp in overlap]),
        boundary=None,
    )

    labeled_chunks = tuple(
        list(img.chunks[:(img.ndim - ndim)])
        + [tuple(cs
                 + (ovp if loc > 0 else 0)
                 + (ovp if loc < len(cs_axis) - 1 else 0)
                 for loc, cs in enumerate(cs_axis))
           for ovp, cs_axis in zip(overlap, img.chunks[-ndim:])
           ]
    )

    block_labeled = da.map_blocks(
        _segmentation_function,
        img_overlapped,
        seg_fn,
        mask,
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
            block_labeled = save_intermediate_array(
                block_labeled,
                filename="temp_labeled.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

        elif persist:
            block_labeled = block_labeled.persist()

    return block_labeled


def _remove_overlapped_labels(labels: da.Array, overlap: List[int],
                              threshold: float = 0.05,
                              ndim: int = 2,
                              relabel: bool = False,
                              persist: Union[bool, pathlib.Path] = False,
                              progressbar: bool = False) -> da.Array:

    block_relabeled = np.empty(labels.numblocks[-ndim:], dtype=object)

    # First, label each block independently, incrementing the labels in that
    # block by the total number of labels from previous blocks. This way, each
    # block's labels are globally unique. Source: dask_image.label
    block_iter = zip(
        np.ndindex(*labels.numblocks[-ndim:]),
        map(functools.partial(operator.getitem, labels),
            da.core.slices_from_chunks(labels.chunks))
    )

    num_chunks = labels.numblocks[-ndim:]
    index, input_block = next(block_iter)
    (block_relabeled[index],
     total) = _delayed_remove_overlapped_objects(
         input_block,
         overlap=overlap,
         threshold=threshold,
         num_chunks=num_chunks,
         chunk_location=tuple([0] * ndim),
         ndim=ndim)

    for index, input_block in block_iter:
        (relabeled_block,
         n) = _delayed_remove_overlapped_objects(input_block,
                                                 overlap=overlap,
                                                 threshold=threshold,
                                                 chunk_location=index,
                                                 num_chunks=num_chunks,
                                                 ndim=ndim)

        if relabel:
            block_label_offset = da.where(relabeled_block > 0, total, 0)
            relabeled_block += block_label_offset
            total += n

        block_relabeled[index] = relabeled_block

    # Put all the blocks together
    block_relabeled = da.block(block_relabeled.tolist())

    if progressbar:
        progressbar_callback = ProgressBar
    else:
        progressbar_callback = Callback

    with progressbar_callback():
        if isinstance(persist, pathlib.Path):
            block_relabeled = save_intermediate_array(
                block_relabeled,
                filename="temp_relabeled.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

        elif persist:
            block_relabeled = block_relabeled.persist()

    return block_relabeled


def _merge_overlapped_tiles(labels: da.Array, overlap: List[int],
                            ndim: int = 2,
                            to_geojson: bool = False,
                            persist: Union[bool, pathlib.Path] = False,
                            progressbar: bool = False) -> da.Array:

    merged_chunks = tuple(
        list(labels.chunks[:(labels.ndim - ndim)])
        + [tuple(cs
                 - (ovp if loc > 0 else 0)
                 - (ovp if loc < len(cs_axis) - 1 else 0)
                 for loc, cs in enumerate(cs_axis))
           for ovp, cs_axis in zip(overlap, labels.chunks[-ndim:])
           ]
    )

    merged_depth = tuple([0] * (labels.ndim - ndim)
                         + [(ovp, ovp) for ovp in overlap])

    # Merge the tiles insto a single labeled image.
    merged_tiles = da.map_overlap(
        _merge_tiles,
        labels,
        overlap=overlap,
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
            merged_tiles = save_intermediate_array(
                merged_tiles,
                filename="temp_merged.zarr",
                out_dir=persist,
                compressor=Blosc(clevel=5)
            )

        elif persist:
            merged_tiles = merged_tiles.persist()

    return merged_tiles


def _dump_merged_tiles(labels: da.Array,
                       overlap: List[int],
                       object_classes: dict,
                       mask: Union[da.Array, None] = None,
                       ndim: int = 2,
                       mask_scale: float = 1.0,
                       out_dir: Union[pathlib.Path, None] = None,
                       persist: Union[bool, pathlib.Path] = False,
                       progressbar: bool = False
                       ) -> Union[da.Array, pathlib.Path]:

    labels_features = da.map_blocks(
        _dump_chunk_geojson,
        labels,
        overlap=overlap,
        object_classes=object_classes,
        out_dir=out_dir,
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
            labels_features = labels_features.persist()

        if mask is not None:
            padded_mask = np.pad(mask, tuple((1, 0) for _ in range(mask.ndim)))
            padded_mask = padded_mask[tuple(slice(None, -1)
                                            for _ in range(mask.ndim))]
            padded_mask += mask

        else:
            mask_scale = 1 / 16
            padded_mask = np.ones([s // 16 for s in labels.shape],
                                  dtype=bool)

        dump_annotations(
            padded_mask,
            object_classes={0: "annotation"},
            filename=out_dir / "annotations.geojson",
            scale=mask_scale,
            offset=None,
            ndim=2,
            keep_all=True
        )

        out_fn = pathlib.Path(str(out_dir) + "-detections.zip")
        with zipfile.ZipFile(out_fn, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=9) as archive:
            for file_path in out_dir.rglob("*.geojson"):
                archive.write(file_path,
                              arcname=file_path.relative_to(out_dir))

        labels_features = out_fn

    return labels_features


def _prepare_input(img: da.Array, ndim: int = 2) -> da.Array:
    chunksize = list(img.chunksize[-ndim:])

    # Prepare input for overlap.
    padding = [(0, 0)] * (img.ndim - ndim)
    padding += [(0, (cs - s) % cs)
                for s, cs in zip(img.shape[-ndim:], chunksize)]

    if any(map(any, padding)):
        img_padded = da.pad(
            img,
            padding
        )

        img_rechunked = da.rechunk(img_padded, chunksize)

    else:
        img_rechunked = img

    return img_rechunked


def _prepare_mask(mask: da.Array, mask_scale: float,
                  chunksize: Union[List[int], Tuple[int]],
                  chunks: Tuple[Tuple[int]],
                  overlap: List[int],
                  ndim: int) -> da.Array:
    if mask is None:
        mask_overlapped = None

    else:
        chunksize_mask = [round(cs * mask_scale) for cs in chunksize]

        mask_rechunked = _prepare_input(
            mask,
            chunksize=chunksize_mask,
            ndim=ndim
        )

        mask_rechunked = da.map_blocks(
            transform.rescale,
            mask_rechunked,
            scale=round(1 / mask_scale),
            order=0,
            chunks=chunks,
            dtype=bool,
            meta=np.empty((0, ), dtype=bool)
        )

        mask_overlapped = da.overlap.overlap(
            mask_rechunked,
            depth=tuple([(ovp, ovp) for ovp in overlap]),
            boundary=None,
        )

    return mask_overlapped


def label(img: da.Array, seg_fn: Callable,
          overlap: Union[int, List[int]] = 50,
          threshold: float = 0.05,
          ndim: int = 2,
          mask: Union[da.Array, None] = None,
          mask_scale: float = 1.0,
          to_geojson: bool = False,
          object_classes: Union[dict, None] = None,
          out_dir: Union[pathlib.Path, str, None] = None,
          persist: Union[bool, pathlib.Path, str] = False,
          progressbar: bool = False,
          segmentation_fn_kwargs: Union[dict, None] = None
          ) -> Union[da.Array, pathlib.Path]:

    if isinstance(overlap, int):
        overlap = [overlap] * ndim

    if out_dir is not None and not isinstance(out_dir, pathlib.Path):
        out_dir = pathlib.Path(out_dir)

    if persist is not None and not isinstance(persist, pathlib.Path):
        persist = pathlib.Path(persist)

    img_rechunked = _prepare_input(img, ndim=ndim)

    mask_overlapped = _prepare_mask(
        mask,
        mask_scale=mask_scale,
        chunksize=img.chunksize[-ndim:],
        chunks=img_rechunked.chunks[-ndim:],
        overlap=overlap,
        ndim=ndim
    )

    labels = _segment_overlapping(
        img_rechunked,
        seg_fn=seg_fn,
        overlap=overlap,
        mask=mask_overlapped,
        ndim=ndim,
        segmentation_fn_kwargs=segmentation_fn_kwargs,
        persist=persist,
        progressbar=progressbar
    )

    labels = _remove_overlapped_labels(
        labels,
        overlap=overlap,
        threshold=threshold,
        ndim=ndim,
        relabel=not to_geojson,
        persist=persist,
        progressbar=progressbar
    )

    labels = _merge_overlapped_tiles(
        labels,
        overlap=overlap,
        ndim=ndim,
        to_geojson=to_geojson,
        persist=persist,
        progressbar=progressbar
    )

    if to_geojson:
        if object_classes is None:
            classes_ids = range(labels.shape[:(labels.ndim - ndim)]
                                if labels.ndim - ndim > 0 else 1)
            object_classes = {class_id: "cell" for class_id in classes_ids}

        labels = _dump_merged_tiles(
            labels,
            overlap=overlap,
            object_classes=object_classes,
            mask=mask_overlapped,
            ndim=ndim,
            mask_scale=mask_scale,
            out_dir=out_dir,
            persist=persist,
            progressbar=progressbar
        )

    return labels
