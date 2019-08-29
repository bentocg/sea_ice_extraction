# Copyright (c) 2019 Bento Goncalves
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__all__ = ['tile_raster']


import rasterio
from rasterio.windows import Window
import numpy as np
import os
from itertools import product


def tile_raster(input_raster: str,
                patch_size: int = 100,
                stride: float = 1,
                thresholds: tuple = (5, 255),
                output_dir: str = 'tiles',
                mask_func = None):
    """
    Creates patches from raster.

    :param input_raster: file name for raster.
    :param patch_size: dimensions for patches.
    :param stride: sliding window stride parametrized as a multiple of patch size.
    :param thresholds: thresholds to throw away blank patches from raster corners.
    :param output_dir: directory where patches will be saved
    :return:
    """
    # read groundtruth, panchromatic and multispectral versions of the scene
    scn_pan = rasterio.open(input_raster)

    # create output folders
    for ele in ['x', 'y']:
        os.makedirs(f"{output_dir}/{ele}", exist_ok=True)

    # template for output filename
    output_filename = output_dir + '/{}/{}_{}_{}.tif'

    # get width and height and prepare iterator for sliding window
    nrows, ncols = scn_pan.shape
    offsets = product(range(0, nrows, int(patch_size * stride)),
                      range(0, ncols, int(patch_size * stride)))
    big_window = Window(col_off=0, row_off=0, width=ncols, height=nrows)

    # extract patches on sliding window
    for row, col in offsets:
        window = Window(col_off=col, row_off=row, width=patch_size,
                        height=patch_size).intersection(big_window)
        patch_pan = scn_pan.read(scn_pan.indexes[0], window=window)

        # check content with thresholds
        if np.max(patch_pan) > thresholds[0] and np.min(
                patch_pan) < thresholds[1] and patch_pan.shape == (patch_size, patch_size):

            try:
                if mask_func is not None:
                    mask = mask_func(patch_pan)
                    with rasterio.open(output_filename.format('y',
                            input_raster.split('/')[-1].split('.')[0], row, col),
                            mode='w',
                            driver='GTiff',
                            width=patch_size,
                            height=patch_size,
                            transform=scn_pan.window_transform(window),
                            crs=scn_pan.crs,
                            count=1,
                            compress='lzw',
                            dtype=rasterio.uint8) as dst:
                        dst.write(mask, indexes=1)

                with rasterio.open(output_filename.format('x',
                        input_raster.split('/')[-1].split('.')[0], row, col),
                                   mode='w',
                                   driver='GTiff',
                                   width=patch_size,
                                   height=patch_size,
                                   transform=scn_pan.window_transform(window),
                                   crs=scn_pan.crs,
                                   count=1,
                                   compress='lzw',
                                   dtype=rasterio.uint8) as dst:
                    dst.write(patch_pan, indexes=1)
            except IOError:
                print(
                    f"({row}, {col}) out of bounds for {input_raster.split('/')[-1].split('.')[0]}")
                continue

    return None
