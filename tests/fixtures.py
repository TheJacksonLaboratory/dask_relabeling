import pytest

import os
import tempfile
import pathlib

import numpy as np
import dask.array as da
from skimage import measure


def segmentation_fun(img: np.ndarray, **kwargs):
    """Simple labeling function to test merging capacity of the code.
    """
    labels = measure.label(img, connectivity=1)
    labels = labels.astype(np.int32)
    return labels


def segmentation_classes_fun(img: np.ndarray, **kwargs):
    """Simple labeling function to test merging capacity of the code in case
    that the segmentation process also computes object classes.
    """
    labels = measure.label(img, connectivity=1)
    labels = labels.astype(np.int32)
    classes = np.where(labels, 1, 0)

    return np.stack((labels, classes))


def load_sample(filename, num_blocks):
    sample_array = np.empty(num_blocks, dtype=object)

    with np.load(filename, "r", allow_pickle=True) as sample_arr_fp:
        for idx in np.ndindex(num_blocks):
            sample_array[idx] = sample_arr_fp["-".join(map(str, idx))]

    sample_array = da.block(sample_array.tolist())
    return sample_array


def input_output_2d():
    num_blocks = (4, 3)

    expected_values = {
        "spatial_dims": 2,
        "segmentation_fun": segmentation_fun,
        "segmentation_fun_kwargs": {},
        "returns_classes": False,
        "object_classes": None,
        "input_arr": load_sample("tests/samples/input_2d.npz", num_blocks),
        "overlapped_input_arr": load_sample("tests/samples/ovp_input_2d.npz",
                                            num_blocks),
        "segmentation_arr": load_sample("tests/samples/seg_2d.npz",
                                        num_blocks),
        "removal_arr": load_sample("tests/samples/rem_2d.npz", num_blocks),
        "trimmed_merged_arr": load_sample("tests/samples/trim_2d.npz",
                                          num_blocks),
        "sorted_merged_arr": load_sample("tests/samples/sort_2d.npz",
                                         num_blocks),
        "overlaps": [2, 2],
        "chunksize": [4, 4],
        "threshold": 0.25,
        "annotations_output": load_sample("tests/samples/ann_2d.npz",
                                          num_blocks),
    }

    return expected_values


def input_output_3d():
    num_blocks = (3, 2, 3)

    expected_values = {
        "spatial_dims": 3,
        "segmentation_fun": segmentation_fun,
        "segmentation_fun_kwargs": {},
        "returns_classes": False,
        "object_classes": None,
        "input_arr": load_sample("tests/samples/input_3d.npz", num_blocks),
        "overlapped_input_arr": load_sample("tests/samples/ovp_input_3d.npz",
                                            num_blocks),
        "segmentation_arr": load_sample("tests/samples/seg_3d.npz",
                                        num_blocks),
        "removal_arr": load_sample("tests/samples/rem_3d.npz", num_blocks),
        "trimmed_merged_arr": load_sample("tests/samples/trim_3d.npz",
                                          num_blocks),
        "sorted_merged_arr": load_sample("tests/samples/sort_3d.npz",
                                         num_blocks),
        "overlaps": [2,  2,  2],
        "chunksize": [5,  5,  5],
        "threshold": 0.125,
        "annotations_output": None
    }

    return expected_values


def add_classes_channel(expected_values):
    expected_values["segmentation_fun"] = segmentation_classes_fun
    expected_values["returns_classes"] = True

    expected_values["segmentation_arr"] = da.stack(
        (expected_values["segmentation_arr"],
            da.where(expected_values["segmentation_arr"], 1, 0)))
    expected_values["segmentation_arr"] = \
        expected_values["segmentation_arr"].rechunk(
            ((2, ), *expected_values["segmentation_arr"].chunks[1:]))

    expected_values["removal_arr"] = da.stack(
        (expected_values["removal_arr"],
            da.where(expected_values["removal_arr"], 1, 0)))
    expected_values["removal_arr"] = \
        expected_values["removal_arr"].rechunk(
            ((2, ), *expected_values["removal_arr"].chunks[1:]))

    expected_values["trimmed_merged_arr"] = da.stack(
        (expected_values["trimmed_merged_arr"],
            da.where(expected_values["trimmed_merged_arr"], 1, 0)))
    expected_values["trimmed_merged_arr"] = \
        expected_values["trimmed_merged_arr"].rechunk(
            ((2, ), *expected_values["trimmed_merged_arr"].chunks[1:]))

    expected_values["sorted_merged_arr"] = da.stack(
        (expected_values["sorted_merged_arr"],
            da.where(expected_values["sorted_merged_arr"], 1, 0)))
    expected_values["sorted_merged_arr"] = \
        expected_values["sorted_merged_arr"].rechunk(
            ((2, ), *expected_values["sorted_merged_arr"].chunks[1:]))

    expected_values["object_classes"] = {1: "cell"}

    return expected_values


@pytest.fixture(scope="module", params=[2, 3])
def input_output(request):
    spatial_dims = request.param

    if spatial_dims == 2:
        expected_values = input_output_2d()
    else:
        expected_values = input_output_3d()

    return expected_values


@pytest.fixture(scope="module", params=[False, True])
def input_output_wclasses(input_output, request):
    expected_values = input_output
    if request.param:
        expected_values = add_classes_channel(expected_values)

    return expected_values


@pytest.fixture(scope="module", params=[False, True])
def input_output_2d_only(request):
    expected_values = input_output_2d()
    if request.param:
        expected_values = add_classes_channel(expected_values)

    return expected_values


@pytest.fixture(scope="module", params=[
    (None, False), (str, True), (pathlib.Path, True), (pathlib.Path, False)
])
def temporal_directory(request):
    temp_dir_class, pre_existent = request.param

    if temp_dir_class is None:
        temporal_dir_name = None

    else:
        temporal_dir = tempfile.TemporaryDirectory()
        temporal_dir_name = temporal_dir.name

        if not pre_existent:
            # Use a non-existent folder inside the temporal directory.
            temporal_dir_name = os.path.join(temporal_dir_name, "temp-images")

        temporal_dir_name = temp_dir_class(temporal_dir_name)

    yield temporal_dir_name

