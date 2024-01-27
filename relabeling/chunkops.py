import pathlib
import itertools

import numpy as np
import scipy
import geojson

from typing import List, Union, Callable

from . import utils


def segmentation_wrapper(img_chunk: np.ndarray, segmentation_fn: Callable,
                         **segmentation_fn_kwargs) -> np.ndarray:
    """Wrapper to apply a segmentation function to each chunk.
    """
    # Execute the segmentation process
    labeled_image = segmentation_fn(img_chunk, **segmentation_fn_kwargs)
    labeled_image = labeled_image.astype(np.int32)

    return labeled_image


def remove_overlapped_objects(labeled_image: np.array, overlaps: List[int],
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


def sort_indices(labeled_image: np.array, ndim: int = 2):
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


def merge_tiles(tile: np.ndarray, overlaps: List[int],
                to_geojson: bool = False,
                block_info: Union[dict, None] = None) -> np.ndarray:
    """Merge objects detected in overlapping regions from adjacent chunks of
    this chunk.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    ndim = tile.ndim

    valid_indices = utils.get_valid_overlaps(chunk_location, num_chunks, ndim)

    # Compute selections from regions to merge
    base_src_sel = tuple(
        map(lambda coord, axis_chunks, axis_overlap:
            slice(axis_overlap if coord > 0 else 0,
                  -axis_overlap if coord < axis_chunks - 1 else None),
            chunk_location, num_chunks, overlaps)
    )

    merging_tile = np.copy(tile[base_src_sel])

    if to_geojson:
        offset_index = np.max(merging_tile)
        offset_labels = np.where(merging_tile, offset_index + 1, 0)
        merging_tile = merging_tile - offset_labels

    left_tile = np.empty_like(merging_tile)

    for indices in valid_indices:
        left_tile[:] = 0
        dst_sel = tuple(map(utils.get_dest_selection,
                            chunk_location, num_chunks, overlaps, indices))
        src_sel = tuple(map(utils.get_source_selection,
                            chunk_location, num_chunks, overlaps, indices))
        left_tile[dst_sel] = tile[src_sel]

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


def compute_features(merged_tile: np.ndarray, overlaps: List[int],
                     object_classes: dict,
                     ndim: int = 2,
                     block_info: Union[dict, None] = None
                     ) -> np.ndarray:
    """Convert detections into GeoJson Feature objects containing the contour
    and class of each detected object in this chunk.
    """
    num_chunks = block_info[None]["num-chunks"]
    chunk_location = block_info[None]["chunk-location"]
    chunk_shape = [
        dim - (axis_overlap if coord > 0 else 0)
        - (axis_overlap if coord < axis_chunks - 1 else 0)
        for dim, coord, axis_chunks, axis_overlap in zip(merged_tile.shape,
                                                         chunk_location,
                                                         num_chunks,
                                                         overlaps)
    ]

    offset = np.array(chunk_location, dtype=np.int64)
    offset *= np.array(chunk_shape)
    offset -= np.array([axis_overlap if coord > 0 else 0
                        for coord, axis_overlap in zip(chunk_location,
                                                       overlaps)])
    offset = offset[[1, 0]]

    detections = utils.labels_to_annotations(
        merged_tile,
        object_classes=object_classes,
        offset=offset,
        ndim=ndim,
        keep_all=False
    )

    merged_tile = np.array([[{"detections": detections}]], dtype=object)

    return merged_tile


def dump_annotation(annotated_tile: np.ndarray, out_dir: pathlib.Path,
                    block_info: Union[dict, None] = None
                    ) -> None:
    filename = (f"detection-"
                f"{'-'.join(map(str, block_info[None]['chunk-location']))}"
                f".geojson")
    filename = out_dir / filename

    annotations = []
    for object_type, cc in annotated_tile[0, 0]["detections"]:
        cc_poly = geojson.Polygon([cc])

        annotations.append(geojson.Feature(geometry=cc_poly))
        annotations[-1]["properties"] = {"objectType": object_type}

    if len(annotations):
        with open(filename, "w") as fp:
            geojson.dump(annotations, fp)

    return np.array([[filename]], dtype=object)
