import pathlib
import itertools

import numpy as np
import scipy
import geojson

from numpy.typing import ArrayLike
from typing import List, Union

from . import utils


def remove_overlapped_objects(labeled_image: ArrayLike, overlaps: List[int],
                              threshold: float = 0.05,
                              ndim: int = 2,
                              block_info: Union[dict, None] = None
                              ) -> np.ndarray:
    """Removes ambiguous objects detected in overlapping regions between
    adjacent chunks.
    """
    # Check whether the segmentation map contains information about the object
    # classes or not.
    if labeled_image.ndim > ndim:
        classes = labeled_image[1]
        labeled_image = labeled_image[0]
    else:
        classes = None

    chunk_location = block_info[None]["chunk-location"]
    num_chunks = block_info[None]["num-chunks"]

    # Remove objects touching the `lower` and `upper` borders of the current
    # axis from labels of this chunk.
    overlapped_labels = []

    for level, (axis, coord, axis_chunks, axis_overlap) in itertools.product(
      [True, False], zip(range(ndim), chunk_location, num_chunks, overlaps)):
        if axis_overlap > 0 and coord > 0 and level:
            in_sel = tuple(
                [slice(None)] * axis
                + [slice(axis_overlap, 2 * axis_overlap)]
                + [slice(None)] * (ndim - 1 - axis)
            )
            out_sel = tuple(
                [slice(None)] * axis
                + [slice(0, axis_overlap)]
                + [slice(None)] * (ndim - 1 - axis)
            )

        elif axis_overlap > 0 and coord < axis_chunks - 1 and not level:
            in_sel = tuple(
                [slice(None)] * axis
                + [slice(-2 * axis_overlap, -axis_overlap)]
                + [slice(None)] * (ndim - 1 - axis)
            )
            out_sel = tuple(
                [slice(None)] * axis
                + [slice(-axis_overlap, None)]
                + [slice(None)] * (ndim - 1 - axis)
            )

        else:
            continue

        in_margin = labeled_image[in_sel]
        out_margin = labeled_image[out_sel]

        for curr_label in np.unique(out_margin):
            if curr_label == 0:
                continue

            # Get the proportion of the current object that is inside and
            # outside the overlap margin.
            in_margin_sum = np.sum(in_margin == curr_label)
            out_margin_sum = np.sum(out_margin == curr_label)
            out_margin_prop = out_margin_sum / (in_margin_sum + out_margin_sum)

            if (coord % 2 != 0 and out_margin_prop >= threshold
               or coord % 2 == 0 and (out_margin_prop > (1.0 - threshold))):
                overlapped_labels.append(curr_label)

    for curr_label in overlapped_labels:
        if classes is not None:
            classes = np.where(labeled_image == curr_label, 0, classes)

        labeled_image = np.where(labeled_image == curr_label, 0, labeled_image)

    if classes is not None:
        labeled_image = np.stack((labeled_image, classes), axis=0)

    return labeled_image


def sort_indices(labeled_image: ArrayLike, ndim: int = 2) -> np.ndarray:
    """Sort label indices to have them as a continuous sequence.
    """
    if labeled_image.ndim > ndim:
        classes = labeled_image[1]
        labeled_image = labeled_image[0]

    else:
        classes = None

    labels_map = np.unique(labeled_image)

    labels_map = scipy.sparse.csr_matrix(
        (np.arange(labels_map.size, dtype=np.int32),
            (labels_map, np.zeros_like(labels_map))),
        shape=(labels_map.max() + 1, 1)
    )

    if ndim == 2:
        labeled_image = labeled_image[None, ...]

    relabeled_image = np.empty_like(labeled_image)

    # Sparse matrices can only be reshaped into 2D matrices, so treat each
    # stack as a different image.
    for label_idx, label_img in enumerate(labeled_image):
        relabel_map = labels_map[label_img.ravel()]
        rlabel_img = relabel_map.reshape(label_img.shape)
        rlabel_img.todense(out=relabeled_image[label_idx, ...])

    if ndim > 2:
        labeled_image = np.stack(relabeled_image, axis=0)

    else:
        labeled_image = relabeled_image[0]

    if classes is not None:
        labeled_image = np.stack((labeled_image, classes), axis=0)

    return labeled_image


def merge_tiles(labeled_image: ArrayLike, overlaps: List[int],
                block_info: Union[dict, None] = None) -> np.ndarray:
    """Merge objects detected in overlapping regions from adjacent chunks of
    this chunk.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    ndim = labeled_image.ndim

    valid_indices = utils.get_valid_overlaps(chunk_location, num_chunks, ndim)

    # Compute selections from regions to merge
    base_src_sel = tuple(
        map(lambda coord, axis_chunks, axis_overlap:
            slice(axis_overlap if coord > 0 else 0,
                  -axis_overlap if coord < axis_chunks - 1 else None),
            chunk_location, num_chunks, overlaps)
    )

    merging_tile = np.copy(labeled_image[base_src_sel])

    left_tile = np.empty_like(merging_tile)

    for indices in valid_indices:
        left_tile[:] = 0
        dst_sel = tuple(map(utils.get_dest_selection,
                            chunk_location, num_chunks, overlaps, indices))
        src_sel = tuple(map(utils.get_source_selection,
                            chunk_location, num_chunks, overlaps, indices))
        left_tile[dst_sel] = labeled_image[src_sel]

        for l_label in np.unique(left_tile):
            if l_label == 0:
                continue

            merging_tile = np.where(
                left_tile == l_label,
                l_label,
                merging_tile
            )

    merging_tile = merging_tile[base_src_sel]

    return merging_tile


def annotate_object_fetures(labeled_image: ArrayLike, overlaps: List[int],
                            object_classes: dict,
                            ndim: int = 2,
                            block_info: Union[dict, None] = None
                            ) -> np.ndarray:
    """Convert detections into GeoJson Feature objects containing the contour
    and class of each detected object in this chunk.
    """
    array_location = block_info[0]['array-location']
    chunk_location = block_info[None]['chunk-location']

    offset_overlaps = list(
        map(lambda coord, axis_overlap:
            2 * coord * axis_overlap,
            chunk_location, overlaps)
    )

    array_location = np.array(array_location, dtype=np.int64)[:, 0]
    offset_overlaps = np.array(offset_overlaps, dtype=np.int64)

    offset = array_location - offset_overlaps
    offset = offset[[1, 0]]

    annotations = utils.labels_to_annotations(
        labeled_image,
        object_classes=object_classes,
        offset=offset,
        ndim=ndim
    )

    labeled_image_annotations = np.array([[annotations]], dtype=object)

    return labeled_image_annotations


def dump_annotaions(labeled_image_annotations: ArrayLike,
                    out_dir: pathlib.Path,
                    block_info: Union[dict, None] = None) -> np.ndarray:

    geojson_annotation = labeled_image_annotations.item()

    if geojson_annotation:
        out_filename = "-".join(map(str, block_info[None]['chunk-location']))
        out_filename += ".geojson"
        out_filename = out_dir / out_filename

        with open(out_filename, "w") as fp:
            geojson.dump(geojson_annotation, fp)

    else:
        out_filename = 0

    return np.array([[out_filename]], dtype=object)