import pytest

import tempfile
import pathlib

import numpy as np
import dask.array as da
from skimage import measure
import zipfile
import geojson

from samples import (CHUNKSIZE, OVERLAPS, THRESHOLD, INPUT_IMG,
                     ANNOTATIONS_OUTPUT, SEGMENTATION_RES, REMOVAL_RES,
                     GLOBAL_SORT_RES, GLOBAL_SORT_RES_OVERLAP,
                     MERGED_OVERLAP_RES, MERGED_OVERLAP_TRIMMED_RES)

from relabel import relabeling


@pytest.fixture
def input_output_2d():
    out_parent_dir = tempfile.TemporaryDirectory(prefix="./tests")
    out_dir = pathlib.Path(out_parent_dir.name) / "test_image"
    persist_dir = pathlib.Path(out_parent_dir.name) / "test_image-temp"

    expected_values = {
        "ndim": 2,
        "input": INPUT_IMG,
        "segmentation_expected": SEGMENTATION_RES,
        "removal_expected": REMOVAL_RES,
        "global_sort_expected": GLOBAL_SORT_RES,
        "global_sorted_overlap_input": GLOBAL_SORT_RES_OVERLAP,
        "merged_overlap_expected": MERGED_OVERLAP_RES,
        "merged_overlap_trimmed_expected": MERGED_OVERLAP_TRIMMED_RES,
        "annotations_output": ANNOTATIONS_OUTPUT,
        "overlaps": OVERLAPS,
        "chunksize": CHUNKSIZE,
        "threshold": THRESHOLD,
        "out_dir": out_dir,
        "persist_dir": persist_dir
    }

    yield expected_values

    out_parent_dir.cleanup()


def segmentation_fun(img: np.ndarray, **kwargs):
    """Simple labeling function to test merging capacity of the code.
    """
    labels = measure.label(img)
    return labels


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


def test_segment_overlapped_input(input_output_2d):
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    input_img = input_output_2d["input"]
    segmentation_expected = input_output_2d["segmentation_expected"]

    input_img_overlapped = relabeling.prepare_input(
        input_img,
        overlaps=overlaps,
        ndim=ndim
    )

    local_sort_output = relabeling.segment_overlapped_input(
        input_img_overlapped,
        seg_fn=segmentation_fun,
        ndim=ndim,
        persist=False,
        progressbar=False
    )

    local_sort_output = local_sort_output.compute()
    segmentation_expected = segmentation_expected.compute()

    assert np.array_equal(local_sort_output, segmentation_expected), \
        (f"Labeled output\n{local_sort_output}\ndoes not match the expected "
         f"segmented image\n{segmentation_expected}")


def test_remove_overlapped_labels(input_output_2d):
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    threshold = input_output_2d["threshold"]

    labels_arr = input_output_2d["segmentation_expected"]
    removal_expected = input_output_2d["removal_expected"]

    local_sort_output = relabeling.remove_overlapped_labels(
        labels=labels_arr,
        overlaps=overlaps,
        threshold=threshold,
        ndim=ndim,
        persist=False,
        progressbar=False
    )

    local_sort_output = local_sort_output.compute()
    removal_expected = removal_expected.compute()

    assert np.array_equal(local_sort_output, removal_expected), \
        (f"Labeled output\n{local_sort_output}\ndoes not match the expected "
         f"labeled image after removal of overlapping regions\n"
         f"{removal_expected}")


def test_sort_overlapped_labels(input_output_2d):
    ndim = input_output_2d["ndim"]

    labels_arr = input_output_2d["removal_expected"]
    sorted_expected = input_output_2d["global_sort_expected"]

    sorted_output = relabeling.sort_overlapped_labels(
        labels=labels_arr,
        ndim=ndim,
        persist=False,
        progressbar=False
    )

    sorted_output = sorted_output.compute()
    sorted_expected = sorted_expected.compute()

    assert np.array_equal(sorted_output, sorted_expected), \
        (f"Labeled output\n{sorted_output}\ndoes not match the expected sorted"
         f" labeled image\n{sorted_expected}")


def test_merge_overlapped_tiles(input_output_2d):
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    labels_arr = input_output_2d["global_sort_expected"]
    merged_expected = input_output_2d["merged_overlap_trimmed_expected"]

    merged_output = relabeling.merge_overlapped_tiles(
        labels=labels_arr,
        overlaps=overlaps,
        ndim=ndim,
        persist=False,
        progressbar=False
    )

    merged_output = merged_output.compute()
    merged_expected = merged_expected.compute()

    assert np.array_equal(merged_output, merged_expected), \
        (f"Labeled output\n{merged_output}\ndoes not match the expected merged"
         f" image\n{merged_expected}")


def test_annotate_labeled_tiles(input_output_2d):
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    labels_input = input_output_2d["removal_expected"]
    annotations_expected = input_output_2d["annotations_output"]

    annotations_output = relabeling.annotate_labeled_tiles(
        labels=labels_input,
        overlaps=overlaps,
        object_classes=None,
        ndim=ndim,
        persist=False,
        progressbar=False
    )

    annotations_expected = annotations_expected.compute()
    annotations_output = annotations_output.compute()

    assert np.all(annotations_output == annotations_expected), f"Different at {np.where(annotations_output != annotations_expected)}:\nexpected={annotations_expected[np.where(annotations_output != annotations_expected)]}\ngot={annotations_output[np.where(annotations_output != annotations_expected)]}"

    assert np.array_equal(annotations_output, annotations_expected), \
        (f"Expected GEOJson annotations to be\n{annotations_expected}\ngot\n"
         f"{annotations_output}")


def test_zip_annotated_labeled_tiles(input_output_2d):
    annotations_input = input_output_2d["annotations_output"]

    out_dir = input_output_2d["out_dir"]

    out_zip_filename = relabeling.zip_annotated_labeled_tiles(
        labels=annotations_input,
        out_dir=out_dir,
        persist=False,
        progressbar=False
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

# TODO: Required tests:
# 1. Test a segmentation tool that also returns cells type
# 2. Test with persist as file
# 3. Test with persist on-the-fly
# 4. Test with objects with holes
