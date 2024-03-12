import os

import numpy as np

import zipfile
import geojson

from .fixtures import *
from relabel import relabeling


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
        (f"Labeled output does not match the expected after removal of overlapping regions\n"
         f"{removal_output[np.where(removal_output != removal_expected)]}\n\n"
         f"{removal_expected[np.where(removal_output != removal_expected)]}")


def test_merge_overlapped_tiles(input_output_wclasses):
    ndim = input_output_wclasses["ndim"]
    overlaps = input_output_wclasses["overlaps"]

    labels_arr = input_output_wclasses["removal_arr"]
    merged_expected = input_output_wclasses["trimmed_merged_arr"]

    merged_output = relabeling.merge_overlapped_tiles(
        labels=labels_arr,
        overlaps=overlaps,
        ndim=ndim
    )

    merged_output = merged_output.compute()
    merged_expected = merged_expected.compute()

    assert np.array_equal(merged_output, merged_expected), \
        (f"Labeled output does not match the expected merging overlapping regions\n"
         f"{merged_output[np.where(merged_output != merged_expected)]}\n\n"
         f"{merged_expected[np.where(merged_output != merged_expected)]}")


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
        (f"Expected GEOJson annotations differ at\n"
         f"{np.where(annotations_expected != annotations_output)}\n\ngot:\n"
         f"{annotations_output[np.where(annotations_output != annotations_expected)]}\n\nexpected:\n"
         f"{annotations_expected[np.where(annotations_output != annotations_expected)]}")


def test_zip_annotated_labeled_tiles(input_output_2d_only, temporal_directory):
    annotations_input = input_output_2d_only["annotations_output"]

    out_zip_filename = relabeling.zip_annotated_labeled_tiles(
        labels=annotations_input,
        out_dir=temporal_directory
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

    if temporal_directory is None:
        os.remove(out_zip_filename)


def test_image2labels(input_output):
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


def test_sort_label_indices(input_output_wclasses):
    ndim = input_output_wclasses["ndim"]

    labels_arr = input_output_wclasses["trimmed_merged_arr"]
    sorted_expected = input_output_wclasses["sorted_merged_arr"]

    sorted_output = relabeling.sort_label_indices(
        labels=labels_arr,
        ndim=ndim
    )

    sorted_output = sorted_output.compute()
    sorted_expected = sorted_expected.compute()

    assert np.array_equal(sorted_output, sorted_expected), \
        (f"Sorted labeled output\n\n{sorted_output}\ndoes not match the "
         f"expected sorted labeled image\n\n{sorted_expected}")
