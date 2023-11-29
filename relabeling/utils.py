import os
import pathlib

import numpy as np
import cv2
import geojson

import dask.array as da
from dask.callbacks import Callback
from dask.diagnostics import ProgressBar
import zarr

from typing import List, Union, Callable


class NoProgressBar(Callback):
    pass


def save_intermediate_array(array: da.Array, filename:Union[pathlib.Path, str],
                            overlap:Union[List[int], None]=None,
                            out_dir:Union[pathlib.Path, str]=".",
                            progressbar:bool=False):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if progressbar:
        progressbar_callbak = ProgressBar
    else:
        progressbar_callbak = NoProgressBar

    if ".zarr" not in filename:
        filename += ".zarr"

    if overlap is not None:
        # Pad the image and rechunk it to have even-sized chunks when
        # saving as Zarr format. Only needed when using to generate GeoJson
        # files, since padding is kept.
        chunksize = array.chunksize

        array = da.pad(
            array,
            tuple(tuple((ovp, 0) for ovp in overlap))
        )
        array = da.rechunk(
            array,
            chunksize
        )

    with progressbar_callbak():
        array.to_zarr(
            os.path.join(out_dir, filename),
            compressor=zarr.Blosc(clevel=5),
            overwrite=True
        )

    loaded_array = da.from_zarr(os.path.join(out_dir, filename))

    if overlap is not None:
        loaded_array = loaded_array[tuple(slice(ovp, None) for ovp in overlap)]

    return loaded_array


def dump_annotations(labels:Union[np.ndarray, None],
                     object_type:str,
                     filename:Union[pathlib.Path, str, None]=None,
                     scale:float=1.0,
                     offset:Union[np.ndarray, None]=None,
                     keep_all=False) -> Union[List[geojson.Feature], str, None]:
    for l in np.unique(labels):
        if l == 0:
            continue
        mask = labels == l
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

        annotations = []
        for p_idx in contours_indices:
            cc = contour_coords[p_idx].squeeze(1)
            if len(cc) < 2:
                continue

            if offset is not None:
                cc += offset[None, :]

            cc = np.vstack((cc, cc[0, None, :])) / scale
        
            cc_poly = geojson.Polygon([cc.tolist()])

            annotations.append(geojson.Feature(geometry=cc_poly))
            annotations[-1]["properties"] = {"objectType": object_type}

    if len(annotations) and filename is not None:
            with open(filename, "w") as fp:
                geojson.dump(annotations, fp)
            
            annotations = filename

    elif len(annotations) == 0:
        annotations = None

    return annotations
