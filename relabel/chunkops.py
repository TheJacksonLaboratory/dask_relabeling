import pathlib
import numpy as np
import json

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
    chunk_location = block_info[None]["chunk-location"]
    num_chunks = block_info[None]["num-chunks"]

    # Remove objects touching the `lower` and `upper` borders of the current
    # axis from labels of this chunk according to the kind of the current
    # chunk.
    chunk_labels = np.unique(labeled_image).tolist()

    in_sel = tuple(
        map(lambda coord, axis_chunks, axis_overlap:
            slice(axis_overlap if coord > 0 else 0,
                  -axis_overlap if coord < axis_chunks - 1 else None),
            chunk_location, num_chunks, overlaps)
    )

    labeled_in_margin = labeled_image[in_sel]

    labeled_in_margin_sum = np.zeros(len(chunk_labels), dtype=np.float32)
    labeled_image_sum = np.zeros(len(chunk_labels), dtype=np.float32)

    map_label2idx = {}
    map_idx2label = {}

    for i, label_id in enumerate(chunk_labels):
        labeled_in_margin_sum[i] = np.sum(labeled_in_margin == label_id)
        labeled_image_sum[i] = np.sum(labeled_image == label_id)
        map_label2idx[label_id] = i
        map_idx2label[i] = label_id

    labeled_in_margin_prop = labeled_in_margin_sum / labeled_image_sum
    labels_region_dim = np.zeros_like(labeled_in_margin_prop, dtype=np.int8)
    labels_region_dim[labeled_in_margin_prop > (1 - threshold)] = ndim + 1
    labels_region_dim[labeled_in_margin_prop < threshold] = -(ndim + 1)

    # Compute the regions to check (faces, edges, and vertices) that are valid
    # overlaps between this chunk and all its adjacent chunks.
    valid_indices = utils.get_valid_overlaps(chunk_location, num_chunks, ndim)

    for indices in valid_indices:
        drop_label = any(map(lambda idx, coord:
                             coord % 2 != 0 if idx is not None else False,
                             indices,
                             chunk_location))

        region_dim = len(list(filter(lambda idx: idx is not None, indices)))

        out_sel = tuple(map(utils.get_source_selection,
                            chunk_location, num_chunks, overlaps, indices))

        labeled_out_margin = labeled_image[out_sel]

        for curr_label in np.unique(labeled_out_margin):
            if curr_label == 0:
                continue

            curr_idx = map_label2idx[curr_label]
            if (labeled_in_margin_prop[curr_idx] >= threshold
               and abs(labels_region_dim[curr_idx]) < region_dim):
                labels_region_dim[curr_idx] = (-1) ** drop_label * region_dim

    # Remove all overlapped objects that are expected to be detected fully by
    # an adjacent chunk.
    overlapped_labels = np.where(labels_region_dim < 0)
    removed_labeled_image = np.copy(labeled_image)

    for label_index in overlapped_labels[0]:
        label_id = map_idx2label[label_index]
        removed_labeled_image = np.where(labeled_image == label_id, 0,
                                         removed_labeled_image)

    # Offset the label indices to prevent collision between indices in
    # different chunks.
    labels_offset = np.ravel_multi_index(chunk_location, num_chunks)
    labels_offset *= 2**31 // np.prod(num_chunks) + 2**31

    labels_offset_array = np.where(removed_labeled_image, labels_offset, 0)

    removed_labeled_image = removed_labeled_image.astype(np.int64)
    removed_labeled_image += labels_offset_array

    return removed_labeled_image


def sort_indices(labeled_image: ArrayLike, unique_labels: list) -> np.ndarray:
    """Sort label indices to have them as a continuous sequence.
    """
    relabeled_image = np.copy(labeled_image)
    for label_id in np.unique(labeled_image):
        relabeled_image = np.where(labeled_image == label_id,
                                   unique_labels.index(label_id),
                                   relabeled_image)

    return relabeled_image


def merge_tiles(labeled_image: ArrayLike, overlaps: List[int],
                ndim: int = 2,
                block_info: Union[dict, None] = None) -> np.ndarray:
    """Merge objects detected in overlapping regions from adjacent chunks of
    this chunk.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    classes = None
    if labeled_image.ndim > ndim:
        classes = labeled_image[1:]
        labeled_image = labeled_image[0]

        num_chunks = num_chunks[1:]
        chunk_location = chunk_location[1:]

    # Compute the regions to check (faces, edges, and vertices) that are valid
    # for this chunk given its location.
    valid_indices = utils.get_merging_overlaps(chunk_location, num_chunks,
                                               ndim)

    # Compute selections from regions to merge
    base_src_sel = tuple(
        map(lambda coord, axis_chunks, axis_overlap:
            slice(axis_overlap if coord > 0 else 0,
                  -axis_overlap if coord < axis_chunks - 1 else None),
            chunk_location, num_chunks, overlaps)
    )

    merging_labeled_image = np.copy(labeled_image[base_src_sel])
    temp_tile_labels = np.empty_like(merging_labeled_image)

    if classes is not None:
        merging_classes = np.copy(classes[(slice(None), *base_src_sel)])
        temp_tile_classes = np.empty_like(merging_classes)

    for indices in valid_indices:
        dst_sel = tuple(map(utils.get_dest_selection,
                            chunk_location, num_chunks, overlaps, indices))
        src_sel = tuple(map(utils.get_source_selection,
                            chunk_location, num_chunks, overlaps, indices))

        temp_tile_labels[:] = 0
        temp_tile_labels[dst_sel] = labeled_image[src_sel]

        if classes is not None:
            temp_tile_classes[:] = 0
            temp_tile_classes[(slice(None), *dst_sel)] = classes[(slice(None),
                                                                  *src_sel)]

        for label_id in np.unique(temp_tile_labels):
            if label_id == 0:
                continue

            labeled_mask = temp_tile_labels == label_id

            merging_labeled_image = np.where(
                labeled_mask,
                label_id,
                merging_labeled_image
            )

            if classes is not None:
                merging_classes = \
                    merging_classes * np.bitwise_not(labeled_mask[None, ...])\
                    + temp_tile_classes * labeled_mask[None, ...]

    if classes is not None:
        merging_labeled_image = np.concatenate(
            (merging_labeled_image[None, ...], merging_classes),
            axis=0
        )

    return merging_labeled_image


def annotate_object_fetures(labeled_image: ArrayLike, overlaps: List[int],
                            object_classes: dict,
                            ndim: int = 2,
                            block_info: Union[dict, None] = None
                            ) -> np.ndarray:
    """Convert detections into GeoJson Feature objects containing the contour
    and class of each detected object in this chunk.
    """
    classes = None
    array_location = block_info[0]['array-location']
    chunk_location = block_info[None]['chunk-location']

    if labeled_image.ndim > ndim:
        classes = labeled_image[1:]
        labeled_image = labeled_image[0]

        array_location = array_location[1:]

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
        classes=classes,
        offset=offset,
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
            json.dump(geojson_annotation, fp)

    else:
        out_filename = 0

    return np.array([[out_filename]], dtype=object)
