import os
import pathlib

import numpy as np
import cv2
import geojson

import dask.array as da

from numcodecs.abc import Codec

from typing import List, Union


def save_intermediate_array(array: da.Array,
                            filename: Union[pathlib.Path, str],
                            out_dir: Union[pathlib.Path, str] = ".",
                            compressor: Codec = None,
                            object_codec: Codec = None):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if ".zarr" not in filename:
        filename += ".zarr"

    chunksize = tuple(
        max(cs_axis)
        for cs_axis in array.chunks
    )

    needs_padding = any(
        any(cs != max_cs for cs in cs_axis)
        for cs_axis, max_cs in zip(array.chunks, chunksize)
    )

    if needs_padding:
        # Pad the image and rechunk it to have even-sized chunks when
        # saving as Zarr format. Only needed when using to generate GeoJson
        # files, since padding is kept.
        padding = tuple(
            (cs_max - cs_axis[0], 0)
            for cs_axis, cs_max in zip(array.chunks, chunksize)
        )

        array = da.pad(
            array,
            padding
        )
        array = da.rechunk(
            array,
            chunksize
        )

    array.to_zarr(
        out_dir / filename,
        compressor=compressor,
        object_codec=object_codec,
        overwrite=True
    )

    loaded_array = da.from_zarr(os.path.join(out_dir, filename))

    if needs_padding:
        loaded_array = loaded_array[tuple(slice(pad[0], None)
                                          for pad in padding)]

    return loaded_array


def labels2annotations(labels: np.ndarray,
                       object_classes: dict,
                       scale: float = 1.0,
                       offset: Union[np.ndarray, None] = None,
                       ndim: int = 2,
                       keep_all: bool = False
                       ) -> Union[List[geojson.Feature], pathlib.Path, None]:

    if labels.ndim > ndim:
        labels = labels[0]
        classes = labels[1]

    else:
        classes = np.zeros_like(labels)

    annotations = []
    for curr_l in np.unique(labels):
        if curr_l == 0:
            continue

        mask = labels == curr_l
        curr_class = np.max(classes[np.nonzero(mask)])
        object_type = object_classes[curr_class]

        (contour_coords,
         hierarchy) = cv2.findContours(mask.astype(np.uint8),
                                       mode=cv2.RETR_TREE,
                                       method=cv2.CHAIN_APPROX_NONE)

        if keep_all:
            contours_indices = np.nonzero(hierarchy[0, :, -1] == -1)[0]
        else:
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

            cc = np.vstack((cc, cc[0, None, :])) / scale

            annotations.append([object_type, cc.tolist()])

    return annotations
