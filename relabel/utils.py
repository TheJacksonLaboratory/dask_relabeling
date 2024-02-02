import os
import pathlib
import itertools

import numpy as np
import cv2
import geojson

import dask.array as da

from numcodecs.abc import Codec

from typing import List, Tuple, Union


def get_valid_overlaps(chunk_location: List[int], num_chunks: List[int],
                       ndim: int) -> List[List[int]]:
    valid_indices = []

    for axis in range(ndim):
        for comb, k in itertools.product(
          itertools.combinations(range(ndim), axis),
          range(2 ** (ndim - axis))):
            indices = list(
                np.unpackbits(np.array([k], dtype=np.uint8), count=ndim - axis,
                              bitorder="little")
            )

            # Insert the fixed indices into the current indices list
            for fixed in comb:
                indices[fixed:fixed] = [None]

            if any(map(lambda level, coord, axis_chunks:
                       level is not None and (
                           coord % 2 == 0
                           or coord == 0 and not level
                           or coord >= axis_chunks - 1 and level),
                       indices, chunk_location, num_chunks)):
                continue

            valid_indices.append(indices)

    return valid_indices


def get_dest_selection(coord: int, axis_chunks: int, axis_overlap: int,
                       axis_level: Union[int, None] = None) -> slice:
    if axis_level is None:
        sel = slice(None)

    elif axis_level:
        sel = slice(-axis_overlap * (2 if coord < axis_chunks - 1 else 1),
                    -axis_overlap if coord < axis_chunks - 1 else None)

    else:
        sel = slice(axis_overlap if coord > 0 else 0,
                    axis_overlap * (2 if coord > 0 else 1))

    return sel


def get_source_selection(coord: int, axis_chunks: int, axis_overlap: int,
                         axis_level: Union[int, None] = None) -> slice:
    if axis_level is None:
        sel = slice(axis_overlap if coord > 0 else None,
                    -axis_overlap if coord < axis_chunks - 1 else None)

    elif axis_level:
        sel = slice(-axis_overlap if coord < axis_chunks - 1 else None, None)

    else:
        sel = slice(0, axis_overlap if coord > 0 else None)

    return sel


def save_intermediate_array(array: da.Array,
                            filename: Union[pathlib.Path, str],
                            out_dir: Union[pathlib.Path, str] = ".",
                            compressor: Codec = None,
                            object_codec: Codec = None
                            ) -> Union[Tuple[Tuple[int]], None]:
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if ".zarr" not in filename:
        filename += ".zarr"

    chunksize = tuple(
        max(cs_axis)
        for cs_axis in array.chunks
    )

    needs_padding = any(
        any(cs != max_cs for cs in cs_axis)
        for cs_axis, max_cs in zip(array.chunks, chunksize)
    )

    if needs_padding:
        # Pad the image and rechunk it to have even-sized chunks when
        # saving as Zarr format. Only needed when using to generate GeoJson
        # files, since padding is kept.
        padding = tuple(
            (cs_max - cs_axis[0], 0)
            for cs_axis, cs_max in zip(array.chunks, chunksize)
        )

        array = da.pad(
            array,
            padding
        )

        array = da.rechunk(
            array,
            chunksize
        )

    else:
        padding = None

    array.to_zarr(
        out_dir / filename,
        compressor=compressor,
        object_codec=object_codec,
        overwrite=True
    )

    return padding


def load_intermediate_array(filename: Union[pathlib.Path, str],
                            padding: Union[List[int], None] = None
                            ) -> da.Array:
    loaded_array = da.from_zarr(filename)

    if padding is not None:
        loaded_array = loaded_array[tuple(slice(pad[0], None)
                                          for pad in padding)]

    return loaded_array


def labels_to_annotations(labels: np.ndarray, object_classes: dict,
                          offset: Union[np.ndarray, None] = None,
                          ndim: int = 2,
                          keep_all: bool = False
                          ) -> Union[List[geojson.Feature], pathlib.Path,
                                     None]:
    if labels.ndim > ndim:
        labels = labels[0]
        classes = labels[1]

    else:
        classes = np.zeros_like(labels)

    annotations = []
    for curr_l in np.unique(labels):
        if curr_l == 0:
            continue

        mask = labels == curr_l
        curr_class = np.max(classes[np.nonzero(mask)])
        object_type = object_classes[curr_class]

        (contour_coords,
         hierarchy) = cv2.findContours(mask.astype(np.uint8),
                                       mode=cv2.RETR_TREE,
                                       method=cv2.CHAIN_APPROX_NONE)

        if keep_all:
            contours_indices = np.nonzero(hierarchy[0, :, -1] == -1)[0]
        else:
            contours_indices = [max(map(lambda i, cc:
                                        (len(cc), i),
                                        range(len(contour_coords)),
                                        contour_coords))[1]]

        for p_idx in contours_indices:
            cc = contour_coords[p_idx].squeeze(1)
            if len(cc) < 2:
                continue

            if offset is not None:
                cc += offset[None, :]

            cc = np.vstack((cc, cc[0, None, :]))

            annotations.append([object_type, cc.tolist()])

    return annotations
