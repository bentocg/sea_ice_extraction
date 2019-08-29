# Copyright (c) 2019 Bento Goncalves
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import rasterio
import geopandas as gpd
from skimage import measure
from utils.footprint_exact import 

# idea
# 1: label regions
# 2: for each region, get area, perimeter and exact_trimmed_geom
# 3: simplify polygons
# 4: get stats (% cover for patch, median floe area, mean floe area, median floe perimeter, mean floe perimeter)
# 5: return or write polygons and stats to main shapefile

def get_exact_trimmed_geom(image, crs=3031, step=48):
    xs, ys = [], []
    ds = rasterio.open(image)
    # get source and target crs
    src_crs = ds.crs
    src_trans = ds.transform
    target_crs = from_epsg(crs)
    # read raster
    inband = ds.read(1).astype(np.uint8)
    nd = ds.nodata
    if nd is None:
        nd = 0
    else:
        inband[inband == nd] = 0
        nd = 0
    height = inband.shape[0]
    pixelst = []
    pixelsb = []
    pts = []
    # For every 'n' line, find first and last data pixel
    lines = list(range(0, height, step))
    try:
        lines_flatnonzero = [np.flatnonzero(inband[l, :] != nd) for l in lines]
    except AttributeError:
        print("Error reading image block.  Check image for corrupt data.")
    i = 0
    for nz in lines_flatnonzero:
        nzmin = nz[0] if nz.size > 0 else 0
        nzmax = nz[-1] if nz.size > 0 else 0
        if nz.size > 0:
            pixelst.append((nzmax + 1, i))
            pixelsb.append((nzmin, i))
        i += step
    pixelsb.reverse()
    pixels = pixelst + pixelsb
    # reproject pixelsdef get_exact_trimmed_geom(image, crs=3031, step=48):
    xs, ys = [], []
    ds = rasterio.open(image)
    # get source and target crs
    src_crs = ds.crs
    src_trans = ds.transform
    target_crs = from_epsg(crs)
    # read raster
    inband = ds.read(1).astype(np.uint8)
    nd = ds.nodata
    if nd is None:
        nd = 0
    else:
        inband[inband == nd] = 0
        nd = 0
    height = inband.shape[0]
    pixelst = []
    pixelsb = []
    pts = []
    # For every 'n' line, find first and last data pixel
    lines = list(range(0, height, step))
    try:
        lines_flatnonzero = [np.flatnonzero(inband[l, :] != nd) for l in lines]
    except AttributeError:
        print("Error reading image block.  Check image for corrupt data.")
    i = 0
    for nz in lines_flatnonzero:
        nzmin = nz[0] if nz.size > 0 else 0
        nzmax = nz[-1] if nz.size > 0 else 0
        if nz.size > 0:
            pixelst.append((nzmax + 1, i))
            pixelsb.append((nzmin, i))
        i += step
    pixelsb.reverse()
    pixels = pixelst + pixelsb
    # reproject pixels
    for px in pixels:
        x, y = src_trans * px
        xs.append(x)
        ys.append(y)
        pts.append((x, y))
    # write polygon (remove redundant vertices)
    geom = Polygon(pts)
    geom = geom.simplify(5)
    # transform crs to target crs
    src_pol = gpd.GeoDataFrame(crs=src_crs, data={'geometry': [geom]}, index=[0])
    target_pol = src_pol.to_crs(target_crs)
    # return geometry
    return mapping(target_pol.geometry)['features'][0]['geometry']


def polygonize(input_raster: str, out_shp: gpd.GeoDataFrame):
    """[summary]
    
    Arguments:
        input_raster {str} -- [description]
        out_shp {gpd.GeoDataFrame} -- [description]
    """
  
    # read input raster
    src = rasterio.open(input_raster)
    affine = src.transform
    patch = src.read(src.indexes[0])

    labels  = measure.label(patch)
    for label in labels:
        patch_lbl = patch * (patch == label)

