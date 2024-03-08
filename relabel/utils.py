import itertools

import numpy as np
import cv2
import geojson

from typing import List, Union


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

            if all(map(lambda level, coord, axis_chunks:
                       (level is None
                        or (coord < axis_chunks - 1 if level else coord > 0)),
                       indices, chunk_location, num_chunks)):
                valid_indices.append(indices)

    return valid_indices


def get_merging_overlaps(chunk_location: List[int], num_chunks: List[int],
                         ndim: int) -> List[List[int]]:
    # Compute all the valid overlaps between the current chunk and all its
    # adjacent chunks.
    merging_indices = get_valid_overlaps(
        chunk_location=chunk_location,
        num_chunks=num_chunks,
        ndim=ndim
    )

    # Merge overlaps in odd-coordinate locations from even-coordinate locations
    merging_indices = list(
        filter(lambda index:
               any(coord % 2 != 0
                   for coord, level in zip(chunk_location, index)
                   if level is not None
                   ),
               merging_indices
               )
    )

    return merging_indices


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


def labels_to_annotations(labels: np.ndarray, object_classes: dict,
                          classes: Union[np.ndarray, None] = None,
                          offset: Union[np.ndarray, None] = None
                          ) -> geojson.Feature:
    annotations = []
    for curr_l in np.unique(labels):
        if curr_l == 0:
            continue

        mask = labels == curr_l
        if classes is not None:
            curr_class = np.max(classes * mask[None, ...])
        else:
            curr_class = 0

        object_type = object_classes[curr_class]

        contour_coords, _ = cv2.findContours(mask.astype(np.uint8),
                                             mode=cv2.RETR_TREE,
                                             method=cv2.CHAIN_APPROX_NONE)

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
            cc_poly = geojson.Polygon([cc.tolist()])

            annotations.append(geojson.Feature(geometry=cc_poly))
            annotations[-1]["properties"] = {"objectType": object_type}

    if not len(annotations):
        annotations = 0

    else:
        annotations = geojson.FeatureCollection(annotations)

    return annotations
