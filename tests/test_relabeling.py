import pytest

import tempfile
import pathlib
import operator

import numpy as np
import dask.array as da
from skimage import measure
import json
import geojson

from samples import (CHUNKSIZE, OVERLAPS, THRESHOLD, INPUT_IMG, OUTPUT_LBL,
                     OVERLAPPED_INPUT, ANNOTATIONS_OUTPUT, FEATRUES_RES,
                     BLOCK_INFOS, SEGMENTATION_RES, REMOVAL_RES,
                     LOCAL_SORT_RES, GLOBAL_SORT_RES, GLOBAL_SORT_RES_OVERLAP,
                     MERGED_OVERLAP_RES, MERGED_OVERLAP_TRIMMED_RES)

from relabel import chunkops, relabeling


@pytest.fixture(scope="module", params=[0, 1, 2, 3, 4, 5, 6, 7, 8])
def input_output_steps_2d(request):
    block_info = BLOCK_INFOS[request.param]
    y, x = block_info[None]["chunk-location"]

    out_dir = tempfile.TemporaryDirectory(prefix="./tests")

    expected_values = {
        "ndim": 2,
        "block_info": block_info,
        "input": OVERLAPPED_INPUT[y][x],
        "segmentation_expected": SEGMENTATION_RES[y][x],
        "removal_expected": REMOVAL_RES[y][x],
        "sorted_expected": GLOBAL_SORT_RES[y][x],
        "global_sorted_overlap_input": GLOBAL_SORT_RES_OVERLAP[y][x],
        "merged_overlap_expected": MERGED_OVERLAP_RES[y][x],
        "merged_overlap_trimmed_expected": MERGED_OVERLAP_TRIMMED_RES[y][x],
        "features_expected": FEATRUES_RES[y][x],
        "annotations_output": ANNOTATIONS_OUTPUT[y][x],
        "overlaps": OVERLAPS,
        "chunksize": CHUNKSIZE,
        "threshold": THRESHOLD,
        "out_dir": pathlib.Path(out_dir.name)
    }

    yield expected_values

    out_dir.cleanup()


@pytest.fixture
def input_output_2d(request):
    out_dir = tempfile.TemporaryDirectory(prefix="./tests")

    input_arr = da.from_array(
        np.array(INPUT_IMG, dtype=np.uint8),
        chunks=CHUNKSIZE
    )

    output_lab = np.array(OUTPUT_LBL, dtype=np.int32)
    segmentation_expected_arr = da.block(SEGMENTATION_RES)
    removal_expected_arr = da.block(REMOVAL_RES)
    global_sort_expected_arr = da.block(GLOBAL_SORT_RES)
    global_sorted_overlap_input_arr = da.block(GLOBAL_SORT_RES_OVERLAP)
    merged_overlap_expected_arr = da.block(MERGED_OVERLAP_RES)
    merged_overlap_trimmed_expected_arr = da.block(MERGED_OVERLAP_TRIMMED_RES)
    features_expected_arr = da.block(FEATRUES_RES)

    with open( pathlib.Path("./tests") / "test_geojson.json", "w") as fp:
        json.dump(ANNOTATIONS_OUTPUT[0][0], fp)

    expected_values = {
        "ndim": 2,
        "input": input_arr,
        "output": output_lab,
        "segmentation_expected": segmentation_expected_arr,
        "removal_expected": removal_expected_arr,
        "global_sort_expected": global_sort_expected_arr,
        "global_sorted_overlap_input": global_sorted_overlap_input_arr,
        "merged_overlap_expected": merged_overlap_expected_arr,
        "merged_overlap_trimmed_expected": merged_overlap_trimmed_expected_arr,
        "features_expected": features_expected_arr,
        "annotations_output": ANNOTATIONS_OUTPUT,
        "overlaps": OVERLAPS,
        "chunksize": CHUNKSIZE,
        "threshold": THRESHOLD,
        "out_dir": pathlib.Path(out_dir.name)
    }

    yield expected_values

    out_dir.cleanup()


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


def test_dump_annotation(input_output_steps_2d):
    block_info = input_output_steps_2d["block_info"]
    features_input = input_output_steps_2d["features_expected"]
    annotations_expected = input_output_steps_2d["annotations_output"]

    out_dir = input_output_steps_2d["out_dir"]

    chunkops.dump_annotation(annotated_tile=features_input,
                             out_dir=out_dir,
                             block_info=block_info)

    out_filename = (
        f"detection-"
        f"{'-'.join(map(str, block_info[None]['chunk-location']))}"
        f".geojson")

    out_filename = out_dir / out_filename

    if len(annotations_expected):
        with open(out_filename, "r") as fp:
            out_geojson_file = geojson.load(fp)

        assert all(map(operator.eq, out_geojson_file, annotations_expected)), \
            (f"Expected dumped GEOJson to be {annotations_expected}, but got "
             f"{out_geojson_file} instead.")
    else:
        assert not out_filename.exists(), \
            (f"Expected no GEOJson be generated for the empty block "
             f"{block_info[None]['chunk-location']}, for "
             f"{annotations_expected}, got {out_filename}")


def test_segment_overlapped_input(input_output_2d):
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    input_img = input_output_2d["input"]
    segmentation_expected = input_output_2d["segmentation_expected"]

    local_sort_output = relabeling.segment_overlapped_input(
        input_img,
        seg_fn=segmentation_fun,
        overlaps=overlaps,
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
        offset_labels=False,
        persist=False,
        progressbar=False
    )

    merged_output = merged_output.compute()
    merged_expected = merged_expected.compute()

    assert np.array_equal(merged_output, merged_expected), \
        (f"Labeled output\n{merged_output}\ndoes not match the expected merged"
         f" image\n{merged_expected}")


def test_merge_features_tiles(input_output_2d):
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    labels_arr = input_output_2d["global_sort_expected"]
    features_expected = input_output_2d["features_expected"]

    features_output = relabeling.merge_features_tiles(
        labels=labels_arr,
        overlaps=overlaps,
        object_classes=None,
        ndim=ndim,
        out_dir=None,
        persist=False,
        progressbar=False
    )

    features_output = features_output.compute()
    features_expected = features_expected.compute()

    check_res = map(
        check_coordinate_list,
        features_output.flatten().tolist(),
        features_expected.flatten().tolist()
    )

    assert all(check_res), \
        (f"Labeled output\n{features_output}\ndoes not match the expected"
         f" merged image\n{features_expected}")


def test_dump_features_tiles(input_output_2d):
    out_dir = input_output_2d["out_dir"]
    ndim = input_output_2d["ndim"]
    overlaps = input_output_2d["overlaps"]

    features_input = input_output_2d["features_expected"]
    annotations_expected = input_output_2d["annotations_output"]

    annotations_output = relabeling.merge_features_tiles(
        features_input,
        overlaps=overlaps,
        object_classes=None,
        ndim=ndim,
        out_dir=out_dir,
        persist=False,
        progressbar=False)

    assert np.array_equal(annotations_output, annotations_expected), \
        (f"Expected geojson features\n{annotations_expected}\n\ndoes not match output features\n{annotations_output}")

    #TODO: Check if can store GeoJSON using the default JSON codec of Zarr