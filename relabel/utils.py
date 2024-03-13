import itertools

import numpy as np
from numpy.typing import ArrayLike

from typing import List, Union, TypeVar, Any

try:
    import cv2

    def find_contours(mask: ArrayLike) -> List[List[int]]:
        contour_coords, _ = cv2.findContours(mask, mode=cv2.RETR_TREE,
                                             method=cv2.CHAIN_APPROX_NONE)
        return contour_coords

except ImportError:

    def find_contours(mask: ArrayLike) -> List[List[int]]:
        # This does not replicates the functionality of findContours, just
        # gives a work around when OpenCV is not installed.
        contour_coords = (np.argwhere(mask)[:, None, (1, 0)], )
        return contour_coords


try:
    import geojson

    FeatureCollection = TypeVar('FeatureCollection', geojson.FeatureCollection,
                                Any)

    def geojson_feature(coordinates_list: List[List[int]]
                        ) -> geojson.Feature:
        feature = geojson.Feature(geometry=geojson.Polygon([coordinates_list]))
        return feature

    def geojson_feature_collection(annotations_list: List[geojson.Feature]
                                   ) -> geojson.FeatureCollection:
        collection = geojson.FeatureCollection(annotations_list)
        return collection

except ImportError:
    # When GeoJSON is not installed, the following functions replicate the
    # basic functionality of that library.
    FeatureCollection = TypeVar('FeatureCollection', dict, Any)

    def geojson_feature(coordinates_list: List[List[float]]) -> dict:
        feature = {
            "geometry": {
                "coordinates": [coordinates_list],
                "type": "Polygon"
            },
            "type": "Feature"
        }
        return feature

    def geojson_feature_collection(annotations_list: List[dict]) -> dict:
        collection = {
            "features": annotations_list,
            "type": "FeatureCollection"
        }
        return collection


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


def labels_to_annotations(labels: ArrayLike, object_classes: dict,
                          classes: Union[ArrayLike, None] = None,
                          offset: Union[ArrayLike, None] = None
                          ) -> FeatureCollection:
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

        contour_coords = find_contours(mask.astype(np.uint8))

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
            annotations.append(geojson_feature(cc.tolist()))
            annotations[-1]["properties"] = {"objectType": object_type}

    if not len(annotations):
        annotations = 0

    else:
        annotations = geojson_feature_collection(annotations)

    return annotations
