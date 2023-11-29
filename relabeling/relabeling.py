import os
import shutil

import numpy as np
import scipy

from skimage import morphology, measure, filters, transform
import cv2

import dask.array as da
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar

from typing import List, Union, Callable
import geojson

import pathlib
import zipfile

import functools
import operator
import itertools


from .utils import save_intermediate_array, dump_annotations


class NoProgressBar(Callback):
    pass


def remove_overlapped_objects(labeled_image: np.array, overlap:List[int],
                              threshold:float,
                              chunk_location: List[int],
                              num_chunks:List[int]):
    ndim = labeled_image.ndim

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
        labeled_image = np.where(labeled_image == mrg_l, 0, labeled_image)

    # Relabel image to have a continuous sequence of indices
    labels_map = np.unique(labeled_image)
    labels_map = scipy.sparse.csr_matrix(
        (np.arange(labels_map.size, dtype=np.int32),
            (labels_map, np.zeros_like(labels_map))),
        shape=(labels_map.max() + 1, 1)
    )

    labels_map = labels_map[labeled_image.ravel()]
    labeled_image = labels_map.reshape(labeled_image.shape).todense()

    return labeled_image


def segmentation_function(img_chunk,
                          segmentation_fn,
                          mask=None,
                          overlap=50,
                          threshold=0.1,
                          ndim=2,
                          block_info=None,
                          **segmentation_fn_kwargs):

    if mask is not None and mask.sum() == 0:
        return np.zeros(img_chunk.shape[-ndim:], dtype=np.int32)

    # Execute the segmentation process
    labeled_image = segmentation_fn(img_chunk, **segmentation_fn_kwargs)
    labeled_image = labeled_image.astype(np.int32)

    if mask is not None:
        labeled_image = np.where(mask, labeled_image, 0)

    labeled_image = remove_overlapped_objects(
        labeled_image,
        overlap,
        threshold,
        block_info[None]["chunk-location"],
        block_info[None]["num-chunks"]
    )

    return labeled_image


def get_overlapping_indices(ndim, chunk_location, num_chunks):
    valid_indices = []
    for d in range(ndim):
        for comb, k in itertools.product(itertools.combinations(range(ndim), d), range(2 ** (ndim - d))):
            indices = list(np.unpackbits(np.array([k], dtype=np.uint8), count=ndim - d, bitorder="little"))
            for fixed in comb:
                indices[fixed:fixed] = [None]

            if any(map(lambda lvl, loc, nchk:
                       lvl is not None and (loc % 2 == 0
                                            or loc == 0 and not lvl
                                            or loc >= nchk - 1 and lvl),
                       indices, chunk_location, num_chunks)):
                continue

            valid_indices.append(indices)

    return valid_indices


def merge_tiles_overlap(tile, overlap=None, dump_geojson=False, block_info=None):
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]

    # Compute selections from regions to merge
    base_src_sel = tuple(
        slice(ovp if loc > 0 else 0, -ovp if loc < nchk - 1 else None)
        for loc, nchk, ovp in zip(chunk_location, num_chunks, overlap)
    )

    merging_tile = np.copy(tile[base_src_sel])

    if dump_geojson:
        offset_index = np.max(merging_tile)
        offset_labels = np.where(merging_tile, offset_index + 1, 0)
        merging_tile = merging_tile - offset_labels

    left_tile = np.empty_like(merging_tile)

    valid_indices = get_overlapping_indices(ndim=tile.ndim, chunk_location=chunk_location, num_chunks=num_chunks)

    valid_src_sel = (
        map(lambda loc, nchk, ovp, level:
            slice(ovp if loc > 0 else 0, -ovp if loc < nchk - 1 else None)
            if level is None else
            slice(0, ovp) if not level else slice(-ovp, None),
            chunk_location, num_chunks, overlap, indices
        )
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
                0 if dump_geojson else l_label,
                merging_tile
            )

    if dump_geojson:
        merged_tile = merging_tile
    else:
        merged_tile = merging_tile[base_src_sel]

    return merged_tile


def dump_chunk_geojson(merged_tile, overlap, object_classes=None, out_dir=None, block_info=None):
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = [
        s - (ovp if loc > 0 else 0) - (ovp if loc < nchk - 1 else 0)
        for s, loc, nchk, ovp in zip(merged_tile.shape, chunk_location, num_chunks, overlap)
    ]

    if object_classes is None:
        object_classes = {
            None: "cell"
        }

    offset = np.array(chunk_location, dtype=np.int64)
    offset *= np.array(chunk_shape)
    offset -= np.array([ovp if loc > 0 else 0
                        for loc, ovp in zip(chunk_location, overlap)])
    offset = offset[[1, 0]]

    out_fn = os.path.join(out_dir, "detections", "detections-" + "-".join(map(str, chunk_location)) + ".geojson")

    # TODO: Pass predicted classes as additional dask.Array, and object
    # types as dictionaries
    detections = dump_annotations(
        merged_tile,
        object_classes[None],
        filename=out_fn,
        scale=1.0,
        offset=offset,
        keep_all=False
    )

    merged_tile = np.array([[detections]], dtype=object)

    return merged_tile


def prepare_input(img: da.Array, chunksize:List[int], ndim:int=2):
    # Prepare input for overlap.
    img_rechunked = da.rechunk(
        img,
        list(img.shape[:img.ndim - ndim]) + chunksize
    )

    padding = [(0, 0)] * (img.ndim - ndim)\
              + [(0, (cs - s) % cs)
                 for s, cs in zip(img.shape[-ndim:], chunksize)
                ]

    if any(map(any, padding)):
        img_padded = da.pad(
            img_rechunked,
            padding
        )

        img_rechunked = da.rechunk(
            img_padded,
            list(img_padded.chunksize[:img.ndim - ndim]) + chunksize
        )

    return img_rechunked


def prepare_mask(mask, mask_scale, chunksize, chunks, overlap, ndim):
    if mask is None:
        mask_overlapped = None

    else:
        chunksize_mask = [round(cs * mask_scale) for cs in chunksize]
        mask_rechunked = prepare_input(
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


def segment_overlapping(img:da.Array, mask:da.Array, seg_fn:Callable,
                        overlap:List[int],
                        ndim:int,
                        segmentation_fn_kwargs:Union[dict, None]=None):

    if segmentation_fn_kwargs is None:
        segmentation_fn_kwargs = {}

    img_overlapped = da.overlap.overlap(
        img,
        depth=tuple([(0, 0)] * (img.ndim - ndim)
                    + [(ovp, ovp) for ovp in overlap]),
        boundary=None,
    )

    block_labeled = da.map_blocks(
        segmentation_function,
        img_overlapped,
        seg_fn,
        mask,
        overlap=overlap,
        threshold=0.05,
        **segmentation_fn_kwargs,
        chunks=img_overlapped.chunks[-ndim:],
        drop_axis=tuple(range(img.ndim - ndim)),
        dtype=np.int32,
        meta=np.empty((0, 0), dtype=np.int32)
    )

    return block_labeled


def relabel_chunks(block_labeled):
    block_relabeled = da.zeros_like(block_labeled, dtype=np.int32)

    total = 0

    n_labels = da.map_blocks(
        np.max,
        block_labeled,
        chunks=(1, 1),
        dtype=np.int32,
        meta=np.empty((0, 0), dtype=np.int32)
    )
    for n, sel in zip(n_labels.ravel(),
                        da.core.slices_from_chunks(block_labeled.chunks)):
        labels_offset = da.where(block_labeled[sel] > 0, total, 0)
        block_relabeled[sel] = block_labeled[sel] + labels_offset
        total += n

    return block_relabeled


def segmenting_tiles(img: da.Array, seg_fn:Callable,
                     mask:Union[da.Array, None]=None,
                     mask_scale:float=1.0,
                     chunksize:Union[int, List[int]]=128,
                     overlap:Union[int, List[int]]=50,
                     ndim:int=2,
                     dump_geojson:bool=False,
                     out_dir:Union[pathlib.Path, str]=None,
                     save_intermediate:bool=False,
                     progressbar:bool=False,
                     segmentation_fn_kwargs:Union[dict, None]=None):

    if isinstance(overlap, int):
        overlap = [overlap] * ndim

    if isinstance(chunksize, int):
        chunksize = [chunksize] * ndim

    if progressbar:
        progressbar_callbak = ProgressBar
    else:
        progressbar_callbak = NoProgressBar

    img_rechunked = prepare_input(img, chunksize=chunksize, ndim=ndim)

    mask_overlapped = prepare_mask(
        mask,
        mask_scale=mask_scale,
        chunksize=chunksize,
        chunks=img_rechunked.chunks[-ndim:],
        overlap=overlap,
        ndim=ndim
    )

    block_labeled = segment_overlapping(
        img_rechunked,
        mask_overlapped,
        seg_fn,
        overlap=overlap,
        ndim=ndim,
        segmentation_fn_kwargs=segmentation_fn_kwargs
    )

    # Intermediate computation of the segmentation. If the segmented image fits
    # in RAM memory, use `save_intermediate=True`, otherwise, save it into a
    # temporary file.
    if save_intermediate:
        block_labeled = save_intermediate_array(
            block_labeled,
            filename="temp_labeled.zarr",
            overlap=[2*ovp for ovp in overlap],
            out_dir=out_dir,
            progressbar=progressbar
        )

    else:
        with progressbar_callbak():
            block_labeled = block_labeled.persist()

    if not dump_geojson:
        block_labeled = relabel_chunks(block_labeled, chunksize, overlap)

        if save_intermediate:
            block_labeled = save_intermediate_array(
                block_labeled,
                filename="temp_relabeled.zarr",
                overlap=[2*ovp for ovp in overlap],
                out_dir=out_dir,
                progressbar=progressbar
            )

            shutil.rmtree(os.path.join(out_dir, "temp_labeled.zarr"))

        else:
            with progressbar_callbak():
                block_labeled = block_labeled.persist()

    # Merge the tiles insto a single labeled image.
    merged_tiles = da.map_overlap(
        merge_tiles_overlap,
        block_labeled,
        overlap=overlap,
        dump_geojson=dump_geojson,
        depth=tuple([(ovp, ovp) for ovp in overlap]),
        boundary=None,
        trim=False,
        chunks=img_rechunked.chunks[-ndim:],
        dtype=np.int32,
        meta=np.empty((0, 0), dtype=np.int32)
    )

    if save_intermediate:
        merged_tiles = save_intermediate_array(
            merged_tiles,
            filename="temp_merged_tiles.zarr",
            overlap=overlap if dump_geojson else None,
            out_dir=out_dir,
            progressbar=progressbar
        )

        if dump_geojson:
            shutil.rmtree(os.path.join(out_dir, "temp_labeled.zarr"))
        else:
            shutil.rmtree(os.path.join(out_dir, "temp_relabeled.zarr"))

    else:
        with progressbar_callbak():
            merged_tiles = merged_tiles.persist()

    if dump_geojson:
        # Merge the tiles insto a single labeled image.
        merged_tiles = da.map_blocks(
            dump_chunk_geojson,
            merged_tiles,
            overlap=overlap,
            out_dir=out_dir,
            chunks=img_rechunked.numblocks[-ndim:],
            dtype=object,
            meta=np.empty((0, 0), dtype=object)
        )

        directory = pathlib.Path(os.path.join(out_dir, "detections"))
        os.makedirs(directory, exist_ok=True)

        with progressbar_callbak():
            merged_tiles = merged_tiles.persist()

        if save_intermediate:
            shutil.rmtree(os.path.join(out_dir, "temp_merged_tiles.zarr"))

        if mask is not None:
            padded_mask = np.pad(mask, tuple((1, 0) for _ in range(mask.ndim)))
            padded_mask = padded_mask[tuple(slice(None, -1)
                                            for _ in range(mask.ndim))]
            padded_mask += mask
        else:
            padded_mask = np.ones(img.shape[-ndim:], dtype=bool)

        if out_dir is not None:
            dump_annotations(
                padded_mask,
                object_type="annotation",
                filename=os.path.join(out_dir, "detections/annotations.geojson"),
                scale=mask_scale,
                offset=None,
                keep_all=True
            )

            out_fn = os.path.join(out_dir, "detections.zip")
            with zipfile.ZipFile(out_fn, "w", zipfile.ZIP_DEFLATED,
                                 compresslevel=9) as archive:
                for file_path in directory.rglob("*.geojson"):
                    archive.write(file_path,
                                  arcname=file_path.relative_to(directory))

            shutil.rmtree(os.path.join(out_dir, "detections"))

            merged_tiles = out_fn

    else:
        # Remove the padding added at the beginning
        img_sel = tuple(slice(0, s) for s in img.shape[-ndim:])
        merged_tiles = merged_tiles[img_sel]
        with progressbar_callbak():
            merged_tiles = merged_tiles.persist()

    return merged_tiles
