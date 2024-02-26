import pytest

import os
import tempfile
import pathlib

import numpy as np
import dask.array as da
from skimage import measure
import zipfile
import geojson

from .samples import (CHUNKSIZE, OVERLAPS, THRESHOLD, INPUT_IMG,
                      OVERLAPPED_INPUT, ANNOTATIONS_OUTPUT, SEGMENTATION_RES,
                      REMOVAL_RES, GLOBALLY_SORTED, OVERLAPPED_GLOBALLY_SORTED,
                      MERGED_OVERLAPPED_GLOBALLY_SORTED, TRIMMED_MERGED_RES)

from .samples3d import (CHUNKSIZE_3D, OVERLAPS_3D, THRESHOLD_3D, INPUT_IMG_3D,
                        OVERLAPPED_INPUT_3D, SEGMENTATION_RES_3D,
                        REMOVAL_RES_3D, GLOBALLY_SORTED_3D,
                        OVERLAPPED_GLOBALLY_SORTED_3D,
                        MERGED_OVERLAPPED_GLOBALLY_SORTED_3D,
                        TRIMMED_MERGED_RES_3D)

from relabel import relabeling


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


def input_output_2d():
    expected_values = {
        "ndim": 2,
        "segmentation_fun": segmentation_fun,
        "segmentation_fun_kwargs": {},
        "returns_classes": False,
        "object_classes": None,
        "input_arr": INPUT_IMG,
        "overlapped_input_arr": OVERLAPPED_INPUT,
        "segmentation_arr": SEGMENTATION_RES,
        "removal_arr": REMOVAL_RES,
        "globally_sorted_arr": GLOBALLY_SORTED,
        "merged_globally_sorted_arr": MERGED_OVERLAPPED_GLOBALLY_SORTED,
        "trimmed_merged_arr": TRIMMED_MERGED_RES,
        "overlaps": OVERLAPS,
        "chunksize": CHUNKSIZE,
        "threshold": THRESHOLD,
        "annotations_output": ANNOTATIONS_OUTPUT
    }

    return expected_values


def input_output_3d():
    expected_values = {
        "ndim": 3,
        "segmentation_fun": segmentation_fun,
        "segmentation_fun_kwargs": {},
        "returns_classes": False,
        "object_classes": None,
        "input_arr": INPUT_IMG_3D,
        "overlapped_input_arr": OVERLAPPED_INPUT_3D,
        "segmentation_arr": SEGMENTATION_RES_3D,
        "removal_arr": REMOVAL_RES_3D,
        "globally_sorted_arr": GLOBALLY_SORTED_3D,
        "merged_globally_sorted_arr": MERGED_OVERLAPPED_GLOBALLY_SORTED_3D,
        "trimmed_merged_arr": TRIMMED_MERGED_RES_3D,
        "overlaps": OVERLAPS_3D,
        "chunksize": CHUNKSIZE_3D,
        "threshold": THRESHOLD_3D,
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

    expected_values["globally_sorted_arr"] = da.stack(
        (expected_values["globally_sorted_arr"],
            da.where(expected_values["globally_sorted_arr"], 1, 0)))
    expected_values["globally_sorted_arr"] = \
        expected_values["globally_sorted_arr"].rechunk(
            ((2, ), *expected_values["globally_sorted_arr"].chunks[1:]))

    expected_values["merged_globally_sorted_arr"] = da.stack(
        (expected_values["merged_globally_sorted_arr"],
            da.where(expected_values["merged_globally_sorted_arr"], 1, 0)))
    expected_values["merged_globally_sorted_arr"] = \
        expected_values["merged_globally_sorted_arr"].rechunk(
            ((2, ),
                *expected_values["merged_globally_sorted_arr"].chunks[1:]))

    expected_values["trimmed_merged_arr"] = da.stack(
        (expected_values["trimmed_merged_arr"],
            da.where(expected_values["trimmed_merged_arr"], 1, 0)))
    expected_values["trimmed_merged_arr"] = \
        expected_values["trimmed_merged_arr"].rechunk(
            ((2, ), *expected_values["trimmed_merged_arr"].chunks[1:]))

    expected_values["object_classes"] = {1: "cell"}

    return expected_values


@pytest.fixture(scope="module", params=[2, 3])
def input_output(request):
    ndim = request.param

    if ndim == 2:
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
def temporal_dir(request):
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


def check_coordinate_list(features_coords_output, features_coords_expected):
    features_coords_output_copy = list(features_coords_output)

    all_match = True
    for coords_exp in features_coords_expected:
        coords_match = True

        for out_id, coords_out in enumerate(features_coords_output_copy):
            coords_match &= coords_exp[0] == coords_out[0]
            coords_match &= any(map(lambda cc_out: cc_out in coords_exp[1],
                                    coords_out[1]))

            if coords_match:
                features_coords_output_copy.pop(out_id)
                break
        else:
            all_match = False

    return all_match


def test_prepare_input(input_output):
    ndim = input_output["ndim"]
    overlaps = input_output["overlaps"]

    input_img = input_output["input_arr"]
    img_overlapped_expected = input_output["overlapped_input_arr"]

    img_overlapped_output = relabeling.prepare_input(
        input_img,
        overlaps=overlaps,
        ndim=ndim
    )

    img_overlapped_output = img_overlapped_output.compute()
    img_overlapped_expected = img_overlapped_expected.compute()

    assert np.array_equal(img_overlapped_output, img_overlapped_expected), \
        (f"Overlapped input array does not match the expected overlapped image"
         f" at\n{np.where(img_overlapped_expected != img_overlapped_output)}\n"
         f"{img_overlapped_expected[np.where(img_overlapped_expected != img_overlapped_output)]}\n\n"
         f"{img_overlapped_output[np.where(img_overlapped_expected != img_overlapped_output)]}")


def test_segment_overlapped_input(input_output_wclasses):
    segmentation_fun = input_output_wclasses["segmentation_fun"]
    ndim = input_output_wclasses["ndim"]
    returns_classes = input_output_wclasses["returns_classes"]

    input_img_overlapped = input_output_wclasses["overlapped_input_arr"]
    segmentation_expected = input_output_wclasses["segmentation_arr"]

    local_sort_output = relabeling.segment_overlapped_input(
        input_img_overlapped,
        seg_fn=segmentation_fun,
        ndim=ndim,
        returns_classes=returns_classes
    )

    local_sort_output = local_sort_output.compute()
    segmentation_expected = segmentation_expected.compute()

    assert np.array_equal(local_sort_output, segmentation_expected), \
        (f"Segmentation output does not match the expected overlapped image "
         f"at\n{np.where(local_sort_output != segmentation_expected)}\n"
         f"{local_sort_output[np.where(local_sort_output != segmentation_expected)]}\n\n"
         f"{segmentation_expected[np.where(local_sort_output != segmentation_expected)]}")


def test_remove_overlapped_labels(input_output_wclasses):
    ndim = input_output_wclasses["ndim"]
    overlaps = input_output_wclasses["overlaps"]
    threshold = input_output_wclasses["threshold"]

    labels_arr = input_output_wclasses["segmentation_arr"]
    removal_expected = input_output_wclasses["removal_arr"]

    removal_output = relabeling.remove_overlapped_labels(
        labels=labels_arr,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim
    )

    removal_output = removal_output.compute()
    removal_expected = removal_expected.compute()

    assert np.array_equal(removal_output, removal_expected), \
        (f"Labeled output does not match the expected after removal of overlapping regions"
         f"at\n{np.where(removal_output != removal_expected)}\n"
         f"{removal_output[np.where(removal_output != removal_expected)]}\n\n"
         f"{removal_expected[np.where(removal_output != removal_expected)]}")


def test_sort_overlapped_labels(input_output_wclasses):
    ndim = input_output_wclasses["ndim"]

    labels_arr = input_output_wclasses["removal_arr"]
    sorted_expected = input_output_wclasses["globally_sorted_arr"]

    sorted_output = relabeling.sort_overlapped_labels(
        labels=labels_arr,
        ndim=ndim
    )

    sorted_output = sorted_output.compute()
    sorted_expected = sorted_expected.compute()

    assert np.array_equal(sorted_output, sorted_expected), \
        (f"Sorted labeled output\n\n{sorted_output}\ndoes not match the "
         f"expected sorted labeled image\n\n{sorted_expected}")


def test_merge_overlapped_tiles(input_output_wclasses):
    ndim = input_output_wclasses["ndim"]
    overlaps = input_output_wclasses["overlaps"]

    labels_arr = input_output_wclasses["globally_sorted_arr"]
    merged_expected = input_output_wclasses["trimmed_merged_arr"]

    merged_output = relabeling.merge_overlapped_tiles(
        labels=labels_arr,
        overlaps=overlaps,
        ndim=ndim
    )

    merged_output = merged_output.compute()
    merged_expected = merged_expected.compute()

    assert np.array_equal(merged_output, merged_expected), \
        (f"Labeled output\n{merged_output}\ndoes not match the expected merged"
         f" image\n{merged_expected}")


def test_annotate_labeled_tiles(input_output_2d_only):
    ndim = input_output_2d_only["ndim"]
    overlaps = input_output_2d_only["overlaps"]
    object_classes = input_output_2d_only["object_classes"]

    labels_input = input_output_2d_only["removal_arr"]
    annotations_expected = input_output_2d_only["annotations_output"]

    annotations_output = relabeling.annotate_labeled_tiles(
        labels=labels_input,
        overlaps=overlaps,
        object_classes=object_classes,
        ndim=ndim
    )

    annotations_expected = annotations_expected.compute()
    annotations_output = annotations_output.compute()

    assert np.array_equal(annotations_output, annotations_expected), \
        (f"Expected GEOJson annotations to be\n{annotations_expected}\ngot\n"
         f"{annotations_output}")


def test_zip_annotated_labeled_tiles(input_output_2d_only, temporal_dir):
    annotations_input = input_output_2d_only["annotations_output"]

    out_zip_filename = relabeling.zip_annotated_labeled_tiles(
        labels=annotations_input,
        out_dir=temporal_dir
    )

    assert str(out_zip_filename).endswith(".zip"), \
        (f"Output filename should be a .zip file, got {out_zip_filename} "
         f"instead.")

    assert out_zip_filename.exists(), \
        (f"Output filename {out_zip_filename} was not generated correctly.")

    annotations_out = np.zeros(annotations_input.shape, dtype=object)

    with zipfile.ZipFile(out_zip_filename) as out_zip:
        for out_file in out_zip.infolist():
            pos = map(int, out_file.filename.split(".geojson")[0].split("-"))

            with out_zip.open(out_file.filename, "r") as out_geojson_fp:
                out_geojson = geojson.load(out_geojson_fp)

            annotations_out[tuple(pos)] = out_geojson

    annotations_input = annotations_input.compute()
    assert np.array_equal(annotations_out, annotations_input), \
        (f"Expected dumped GEOJson zip file to be {annotations_input}, but got"
         f" {annotations_out} instead.")

    if temporal_dir is None:
        os.remove(out_zip_filename)


def test_image2labels(input_output, temporal_dir):
    segmentation_fun = input_output["segmentation_fun"]
    segmentation_fun_kwargs = input_output["segmentation_fun_kwargs"]
    returns_classes = input_output["returns_classes"]

    ndim = input_output["ndim"]
    overlaps = input_output["overlaps"]
    threshold = input_output["threshold"]

    input_img = input_output["input_arr"]
    labels_expected = input_output["trimmed_merged_arr"]

    labels_output = relabeling.image2labels(
        input_img,
        seg_fn=segmentation_fun,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        returns_classes=returns_classes,
        temp_dir=temporal_dir,
        segmentation_fn_kwargs=segmentation_fun_kwargs
    )

    labels_expected = labels_expected.compute()
    labels_output = labels_output.compute()

    assert np.array_equal(labels_output, labels_expected), \
        (f"Expected labels to be\n{labels_expected}\ngot\n"
         f"{labels_output}")


def test_image2geojson(input_output_2d_only):
    segmentation_fun = input_output_2d_only["segmentation_fun"]
    segmentation_fun_kwargs = input_output_2d_only["segmentation_fun_kwargs"]
    returns_classes = input_output_2d_only["returns_classes"]

    ndim = input_output_2d_only["ndim"]
    overlaps = input_output_2d_only["overlaps"]
    threshold = input_output_2d_only["threshold"]

    object_classes = input_output_2d_only["object_classes"]

    input_img = input_output_2d_only["input_arr"]
    annotations_expected = input_output_2d_only["annotations_output"]

    annotations_output = relabeling.image2geojson(
        input_img,
        seg_fn=segmentation_fun,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        returns_classes=returns_classes,
        object_classes=object_classes,
        segmentation_fn_kwargs=segmentation_fun_kwargs
    )

    annotations_expected = annotations_expected.compute()
    annotations_output = annotations_output.compute()

    assert np.array_equal(annotations_output, annotations_expected), \
        (f"Expected GEOJson annotations to be\n{annotations_expected}\ngot\n"
         f"{annotations_output}")


def test_labels2geojson(input_output_2d_only):
    ndim = input_output_2d_only["ndim"]
    overlaps = input_output_2d_only["overlaps"]
    threshold = input_output_2d_only["threshold"]

    object_classes = input_output_2d_only["object_classes"]

    labeled_input_img = input_output_2d_only["trimmed_merged_arr"]
    annotations_expected = input_output_2d_only["annotations_output"]

    annotations_output = relabeling.labels2geojson(
        labeled_input_img,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        object_classes=object_classes,
        pre_overlapped=False
    )

    annotations_expected = annotations_expected.compute()
    annotations_output = annotations_output.compute()

    assert np.array_equal(annotations_output, annotations_expected), \
        (f"Expected GEOJson annotations to be\n{annotations_expected}\ngot\n"
         f"{annotations_output}")
