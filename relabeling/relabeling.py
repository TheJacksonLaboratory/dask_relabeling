from typing import List, Union, Callable
import numpy as np
import dask.array as da
import functools


def segmentation_function_decorator(segmentation_fn:Callable):
    """ Wraps the ambiguous labels removal process around the segmentation
    function.
    """
    def segmentation_function(img_chunk, overlap: List[int],
                              threshold:float=0.05,
                              block_info:dict=None,
                              **segmentation_fn_kwargs):
        # Execute the segmentation function.
        labeled_image = segmentation_fn(img_chunk, **segmentation_fn_kwargs)
        labeled_image = labeled_image.astype(np.int32)
        ndim = labeled_image.ndim

        chunk_location = block_info[None]["chunk-location"]
        num_chunks = block_info[None]["num-chunks"]

        tile_base_sel = [slice(None)] * ndim

        for i, (loc, nchk, ovp) in enumerate(zip(chunk_location, num_chunks,
                                                 overlap)):
            if loc > 0:
                # Remove objects touching the `lower` border of the current
                # axis from labels of this chunk.
                mrg_sel = list(tile_base_sel)

                mrg_sel[i] = slice(0, ovp)
                out_margin = labeled_image[tuple(mrg_sel)]

                mrg_sel[i] = slice(ovp, 2 * ovp)
                in_margin = labeled_image[tuple(mrg_sel)]

                margin_labels = np.unique(out_margin)

                for mrg_l in margin_labels:
                    if mrg_l == 0:
                        continue

                    in_mrg = np.sum(in_margin == mrg_l)
                    out_mrg = np.sum(out_margin == mrg_l)
                    out_mrg_prop = out_mrg / (in_mrg + out_mrg)
                    in_mrg_prop = 1.0 - out_mrg_prop

                    if (loc % 2 != 0 and out_mrg_prop >= threshold
                      or loc % 2 == 0 and in_mrg_prop < threshold):
                        labeled_image[np.where(labeled_image == mrg_l)] = 0
                    
            if loc < nchk - 1:
                # Remove objects touching the `upper` border of the current
                # axis from labels of this chunk.
                mrg_sel = list(tile_base_sel)

                mrg_sel[i] = slice(-ovp, None)
                out_margin = labeled_image[tuple(mrg_sel)]

                mrg_sel[i] = slice(-2 * ovp, -ovp)
                in_margin = labeled_image[tuple(mrg_sel)]

                margin_labels = np.unique(out_margin)

                for mrg_l in margin_labels:
                    if mrg_l == 0:
                        continue

                    in_mrg = np.sum(in_margin == mrg_l)
                    out_mrg = np.sum(out_margin == mrg_l)
                    out_mrg_prop = out_mrg / (in_mrg + out_mrg)
                    in_mrg_prop = 1.0 - out_mrg_prop

                    if (loc % 2 != 0 and out_mrg_prop >= threshold
                      or loc % 2 == 0 and in_mrg_prop < threshold):
                        labeled_image[np.where(labeled_image == mrg_l)] = 0

        return labeled_image

    return segmentation_function


def merge_tiles_overlap(tile:np.ndarray, overlap:List[int],
                        block_info:Union[dict, None]=None):
    """Merges the labeled objects from adjacent chunks into this one.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = block_info[None]["chunk-shape"]

    # Compute selections from overlapped regions to merge into this chunk.
    base_src_sel = tuple(
        slice(ovp * (2 if loc > 0 else 0),
              ovp * (2 if loc > 0 else 0) + s)
        for s, loc, ovp in zip(chunk_shape, chunk_location, overlap)
    )

    src_sel_list = [
        tuple(
            [slice(ovp * (2 if loc_j > 0 else 0),
                   ovp * (2 if loc_j > 0 else 0) + s)
             for s, loc_j, ovp in zip(chunk_shape[:i], chunk_location[:i],
                                      overlap[:i])]
            + [slice(0, overlap[i])]
            + [slice(ovp * (2 if loc_j > 0 else 0),
                     ovp * (2 if loc_j > 0 else 0) + s)
               for s, loc_j, ovp in zip(chunk_shape[i+1:],
                                        chunk_location[i+1:],
                                        overlap[i+1:])]
        )
        for i, loc in enumerate(chunk_location)
        if 0 < loc and loc % 2
    ]

    src_sel_list += [
        tuple(
            [slice(ovp * (2 if loc_j > 0 else 0),
                   ovp * (2 if loc_j > 0 else 0) + s)
             for s, loc_j, ovp in zip(chunk_shape[:i], chunk_location[:i],
                                      overlap[:i])]
            + [slice(-overlap[i], None)]
            + [slice(ovp * (2 if loc_j > 0 else 0),
                     ovp * (2 if loc_j > 0 else 0) + s)
               for s, loc_j, ovp in zip(chunk_shape[i+1:],
                                        chunk_location[i+1:],
                                        overlap[i+1:])]
        )
        for i, (loc, nchk) in enumerate(zip(chunk_location, num_chunks))
        if loc < nchk - 1 and loc % 2
    ]

    dst_sel_list = [
        tuple(
            [slice(None)] * i + [slice(0, overlap[i])]
            + [slice(None)] * (tile.ndim - i - 1)
        )
        for i, loc in enumerate(chunk_location)
        if loc > 0 and loc % 2
    ]

    dst_sel_list += [
        tuple(
            [slice(None)] * i + [slice(-overlap[i], None)]
            + [slice(None)] * (tile.ndim - i - 1)
        )
        for i, (loc, nchk) in enumerate(zip(chunk_location, num_chunks))
        if loc < nchk - 1 and loc % 2
    ]

    for d in range(2 ** tile.ndim):
        corner = np.unpackbits(
            np.array((d), dtype=np.uint8),
            count=tile.ndim,
            bitorder="little"
        )

        # Only check valid corners.
        if any(loc >= nchk - 1 if c_idx else loc == 0
               for c_idx, loc, nchk in zip(corner, chunk_location,num_chunks)
               ):
            continue

        corner_slice = tuple(
            slice(-ovp, None) if c_idx else slice(0, ovp)
            for c_idx, ovp in zip(corner, overlap)
        )

        src_sel_list.append(corner_slice)
        dst_sel_list.append(corner_slice)

    # Merge center tile region, that is not overlapped by any adjacent chunk.
    merged_tile = np.copy(tile[base_src_sel])
    left_tile = np.zeros_like(merged_tile)

    # Merge into the current tile the overlapping regions from adjacent chunks.
    for src_sel, dst_sel in zip(src_sel_list, dst_sel_list):
        left_tile[dst_sel] = tile[src_sel]

        for l_label in np.unique(left_tile)[1:]:
            left_mask = left_tile == l_label
            merged_tile[np.nonzero(left_mask)] = l_label

    return merged_tile


def compute_checkered_positions(num_blocks:List[int]):
    """Compute the positions in a checkered pattern that is used to determine
    the order in which to apply the segmentation function.
    """
    ndim = len(num_blocks)
    coords = np.indices(num_blocks)
    coords_even = functools.reduce(
        np.bitwise_and,
        map(lambda c_ax: c_ax % 2 == 0, coords)
    )
    coords_even = np.stack(np.nonzero(coords_even)).T

    coords_offsets = np.stack([
        np.unpackbits(np.array((d, ), dtype=np.uint8), count=ndim,
                      bitorder="little")
        for d in range(2**ndim)
    ])

    checkered_coords = coords_even[None, ...] + coords_offsets[:, None, :]
    checkered_indices = []
    for chk_coords in checkered_coords:
        valid_chk_coords = chk_coords[np.all(chk_coords < np.array(num_blocks),
                                             axis=1)]
        checkered_indices.append(
            [np.ravel_multi_index(c, num_blocks) for c in valid_chk_coords]
        )

    return checkered_indices


def segmenting_tiles(img: da.Array, seg_fn:Callable,
                     chunksize:Union[List[int],int]=128,
                     overlap:Union[List[int], int]=50,
                     ndim:int=2,
                     compute_intermediate:bool=True,
                     **segmentation_fn_kwargs):
    """Performs the chunk-wise segmentation of input `img` using function
    `seg_fn` and merges all the labeled objects into a single labeled image.
    """
    if isinstance(overlap, int):
        overlap = [overlap] * ndim

    if isinstance(chunksize, int):
        chunksize = [chunksize] * ndim

    # Wrap the segmentation function with the decorator.
    block_seg_fn = segmentation_function_decorator(seg_fn)

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

    out_chunks = tuple(
        tuple(cs + (ovp if loc > 0 else 0) + (ovp if loc < nchk-1 else 0)
              for loc, cs in enumerate(cs_ax)
              )
        for cs_ax, nchk, ovp in zip(img_rechunked.chunks[-ndim:],
                                    img_rechunked.numblocks[-ndim:],
                                    overlap)
    )

    # We can use map_blocks because the input is already overlapped.
    block_labeled = da.map_overlap(
        block_seg_fn,
        img_rechunked,
        **segmentation_fn_kwargs,
        overlap=overlap,
        threshold=0.05,
        chunks=out_chunks,
        depth=tuple([(0, 0)] * (img.ndim - ndim)
                    + [(ovp, ovp) for ovp in overlap]),
        boundary=None,
        trim=False,
        drop_axis=(range(img.ndim - ndim)),
        dtype=np.int64,
        meta=np.empty((0, ), dtype=np.int64)
    )

    # Intermediate computation of the segmentation. Because it could fit in the
    # RAM memory, it can be run and be keept for following merging process.
    if compute_intermediate:
        block_labeled = block_labeled.persist()

    checkered_indices = compute_checkered_positions(block_labeled.numblocks)
    sel_coords = da.core.slices_from_chunks(block_labeled.chunks)

    total = 0

    merged_tiles_offset = da.zeros(img_rechunked.shape[-ndim:],
                                   chunks=img_rechunked.chunks[-ndim:],
                                   dtype=np.int32)

    for chk_idx in checkered_indices:
        block_relabeled = da.zeros_like(block_labeled)

        for c_idx in chk_idx:
            sel = sel_coords[c_idx]
            n = da.max(block_labeled[sel])

            # Add the labels offset, so there is a continuous sequence of label
            # indices across the segmented image.
            labels_offset = da.where(block_labeled[sel] > 0, total, 0)
            block_relabeled[sel] = block_labeled[sel] + labels_offset

            total += n

        # Merge the tiles into a single labeled image.
        merged_tiles_offset += da.map_overlap(
            merge_tiles_overlap,
            block_relabeled,
            overlap=overlap,
            depth=tuple([(ovp, ovp) for ovp in overlap]),
            boundary=None,
            trim=False,
            chunks=img_rechunked.chunks[-ndim:],
            dtype=np.int64,
            meta=np.empty((0, ), dtype=np.int64)
        )

    # Remove the padding added at the beginning
    img_sel = tuple(slice(0, s) for s in img.shape[-ndim:])
    merged_tiles = merged_tiles_offset[img_sel]

    return merged_tiles
