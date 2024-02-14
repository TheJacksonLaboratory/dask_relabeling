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
                        TRIMMED_MERGED_RES_3D)

from relabel import relabeling


def segmentation_fun(img: np.ndarray, **kwargs):
    """Simple labeling function to test merging capacity of the code.
    """
    labels = measure.label(img)
    labels = labels.astype(np.int32)
    return labels


def segmentation_3d_fun(img: np.ndarray, **kwargs):
    """Simple labeling function to test merging capacity of the code.
    """
    labels = measure.label(img)
    labels = labels.astype(np.int32)
    return labels


def segmentation_classes_fun(img: np.ndarray, **kwargs):
    """Simple labeling function to test merging capacity of the code in case
    that the segmentation process also computes object classes.
    """
    labels = measure.label(img)
    labels = labels.astype(np.int32)
    classes = np.where(labels, 1, 0)

    return np.stack((labels, classes))


@pytest.fixture(scope="module", params=[2, 3])
def input_output(request):
    if request.param == 2:
        expected_values = {
            "ndim": 2,
            "returns_classes": False,
            "segmentation_fun": segmentation_fun,
            "segmentation_fun_kwargs": {},
            "object_classes": None,
            "input": INPUT_IMG,
            "trimmed_merged_res": TRIMMED_MERGED_RES,
            "overlaps": OVERLAPS,
            "chunksize": CHUNKSIZE,
            "threshold": THRESHOLD,
        }
    else:
        expected_values = {
            "ndim": 3,
            "returns_classes": False,
            "segmentation_fun": segmentation_3d_fun,
            "segmentation_fun_kwargs": {},
            "object_classes": None,
            "input": INPUT_IMG_3D,
            "trimmed_merged_res": TRIMMED_MERGED_RES_3D,
            "overlaps": OVERLAPS_3D,
            "chunksize": CHUNKSIZE_3D,
            "threshold": THRESHOLD_3D,
        }

    yield expected_values


@pytest.fixture(scope="module", params=[True, False])
def input_output_steps_2d(request):
    returns_classes = request.param

    segmentation_res_arr = SEGMENTATION_RES
    removal_res_arr = REMOVAL_RES
    globally_sorted_arr = GLOBALLY_SORTED
    over_glob_sorted_arr = OVERLAPPED_GLOBALLY_SORTED
    merged_over_glob_sorted_arr = MERGED_OVERLAPPED_GLOBALLY_SORTED
    trimmed_merged_res_arr = TRIMMED_MERGED_RES

    if returns_classes:
        seg_fun = segmentation_classes_fun

        segmentation_res_arr = da.stack(
            (segmentation_res_arr,
             da.where(segmentation_res_arr, 1, 0)))
        segmentation_res_arr = segmentation_res_arr.rechunk(
            ((2, ), *segmentation_res_arr.chunks[1:])
        )

        removal_res_arr = da.stack(
            (removal_res_arr,
             da.where(removal_res_arr, 1, 0)))
        removal_res_arr = removal_res_arr.rechunk(
            ((2, ), *removal_res_arr.chunks[1:])
        )

        globally_sorted_arr = da.stack(
            (globally_sorted_arr,
             da.where(globally_sorted_arr, 1, 0)))
        globally_sorted_arr = globally_sorted_arr.rechunk(
            ((2, ), *globally_sorted_arr.chunks[1:])
        )

        over_glob_sorted_arr = da.stack(
            (over_glob_sorted_arr,
             da.where(over_glob_sorted_arr, 1, 0)))
        over_glob_sorted_arr = over_glob_sorted_arr.rechunk(
            ((2, ), *over_glob_sorted_arr.chunks[1:])
        )

        merged_over_glob_sorted_arr = da.stack(
            (merged_over_glob_sorted_arr,
             da.where(merged_over_glob_sorted_arr, 1, 0)))
        merged_over_glob_sorted_arr = merged_over_glob_sorted_arr.rechunk(
            ((2, ), *merged_over_glob_sorted_arr.chunks[1:])
        )

        trimmed_merged_res_arr = da.stack(
            (trimmed_merged_res_arr,
             da.where(trimmed_merged_res_arr, 1, 0)))
        trimmed_merged_res_arr = trimmed_merged_res_arr.rechunk(
            ((2, ), *trimmed_merged_res_arr.chunks[1:])
        )

        object_classes = {1: "cell"}

    else:
        seg_fun = segmentation_fun
        object_classes = None

    expected_values = {
        "ndim": 2,
        "returns_classes": returns_classes,
        "segmentation_fun": seg_fun,
        "segmentation_fun_kwargs": {},
        "object_classes": object_classes,
        "input": INPUT_IMG,
        "overlapped_input": OVERLAPPED_INPUT,
        "segmentation_res": segmentation_res_arr,
        "removal_res": removal_res_arr,
        "globally_sorted": globally_sorted_arr,
        "overlapped_globally_sorted": over_glob_sorted_arr,
        "merged_overlapped_globally_sorted": merged_over_glob_sorted_arr,
        "trimmed_merged_res": trimmed_merged_res_arr,
        "annotations_output": ANNOTATIONS_OUTPUT,
        "overlaps": OVERLAPS,
        "chunksize": CHUNKSIZE,
        "threshold": THRESHOLD,
    }

    yield expected_values


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


def test_prepare_input(input_output_steps_2d):
    ndim = input_output_steps_2d["ndim"]
    overlaps = input_output_steps_2d["overlaps"]

    input_img = input_output_steps_2d["input"]
    img_overlapped_expected = input_output_steps_2d["overlapped_input"]

    img_overlapped_output = relabeling.prepare_input(
        input_img,
        overlaps=overlaps,
        ndim=ndim
    )

    img_overlapped_output = img_overlapped_output.compute() / 255
    img_overlapped_expected = img_overlapped_expected.compute() / 255

    assert np.array_equal(img_overlapped_output, img_overlapped_expected), \
        (f"Overlapped input\n{img_overlapped_output}\ndoes not match the "
         f"expected overlapped image\n{img_overlapped_expected}")


def test_segment_overlapped_input(input_output_steps_2d):
    segmentation_fun = input_output_steps_2d["segmentation_fun"]
    ndim = input_output_steps_2d["ndim"]
    returns_classes = input_output_steps_2d["returns_classes"]

    input_img_overlapped = input_output_steps_2d["overlapped_input"]
    segmentation_expected = input_output_steps_2d["segmentation_res"]

    local_sort_output = relabeling.segment_overlapped_input(
        input_img_overlapped,
        seg_fn=segmentation_fun,
        ndim=ndim,
        returns_classes=returns_classes
    )

    local_sort_output = local_sort_output.compute()
    segmentation_expected = segmentation_expected.compute()

    assert np.array_equal(local_sort_output, segmentation_expected), \
        (f"Segmentation output\n{local_sort_output}\ndoes not match the "
         f"expected labeled image\n{segmentation_expected}")


def test_remove_overlapped_labels(input_output_steps_2d):
    ndim = input_output_steps_2d["ndim"]
    overlaps = input_output_steps_2d["overlaps"]
    threshold = input_output_steps_2d["threshold"]

    labels_arr = input_output_steps_2d["segmentation_res"]
    removal_expected = input_output_steps_2d["removal_res"]

    removal_output = relabeling.remove_overlapped_labels(
        labels=labels_arr,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim
    )

    removal_output = removal_output.compute()
    removal_expected = removal_expected.compute()

    assert np.array_equal(removal_output, removal_expected), \
        (f"Labeled output\n\n{removal_output}\ndoes not match the expected "
         f"labeled image after removal of overlapping regions\n"
         f"{removal_expected}")


def test_sort_overlapped_labels(input_output_steps_2d):
    ndim = input_output_steps_2d["ndim"]

    labels_arr = input_output_steps_2d["removal_res"]
    sorted_expected = input_output_steps_2d["globally_sorted"]

    sorted_output = relabeling.sort_overlapped_labels(
        labels=labels_arr,
        ndim=ndim
    )

    sorted_output = sorted_output.compute()
    sorted_expected = sorted_expected.compute()

    assert np.array_equal(sorted_output, sorted_expected), \
        (f"Sorted labeled output\n\n{sorted_output}\ndoes not match the "
         f"expected sorted labeled image\n\n{sorted_expected}")


def test_merge_overlapped_tiles(input_output_steps_2d):
    ndim = input_output_steps_2d["ndim"]
    overlaps = input_output_steps_2d["overlaps"]

    labels_arr = input_output_steps_2d["globally_sorted"]
    merged_expected = input_output_steps_2d["trimmed_merged_res"]

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


def test_annotate_labeled_tiles(input_output_steps_2d):
    ndim = input_output_steps_2d["ndim"]
    overlaps = input_output_steps_2d["overlaps"]
    object_classes = input_output_steps_2d["object_classes"]

    labels_input = input_output_steps_2d["removal_res"]
    annotations_expected = input_output_steps_2d["annotations_output"]

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


def test_zip_annotated_labeled_tiles(input_output_steps_2d, temporal_dir):
    annotations_input = input_output_steps_2d["annotations_output"]

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

    input_img = input_output["input"]
    labels_expected = input_output["trimmed_merged_res"]

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


def test_image2geojson(input_output_steps_2d):
    segmentation_fun = input_output_steps_2d["segmentation_fun"]
    segmentation_fun_kwargs = input_output_steps_2d["segmentation_fun_kwargs"]
    returns_classes = input_output_steps_2d["returns_classes"]

    ndim = input_output_steps_2d["ndim"]
    overlaps = input_output_steps_2d["overlaps"]
    threshold = input_output_steps_2d["threshold"]

    object_classes = input_output_steps_2d["object_classes"]

    input_img = input_output_steps_2d["input"]
    annotations_expected = input_output_steps_2d["annotations_output"]

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


def test_labels2geojson(input_output_steps_2d):
    import matplotlib.pyplot as plt

    ndim = input_output_steps_2d["ndim"]
    overlaps = input_output_steps_2d["overlaps"]
    threshold = input_output_steps_2d["threshold"]

    object_classes = input_output_steps_2d["object_classes"]

    labeled_input_img = input_output_steps_2d["trimmed_merged_res"]
    annotations_expected = input_output_steps_2d["annotations_output"]

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
