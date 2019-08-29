# Copyright (c) 2019 Bento Goncalves
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


import os
import time
from itertools import product
from argparse import ArgumentParser
from shapely.geometry import mapping

import geopandas as gpd
import rasterio
import fiona
from fiona.crs import from_epsg
from rasterio.windows import Window
from pyproj import transform
from scipy.sparse.csgraph import connected_components
from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import cpu_count
from collections import namedtuple

from utils.extract_sea_ice import extract_sea_ice
from utils.polygonize_raster import polygonize_raster


# Define argument parser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="input directory containing rasters")
    parser.add_argument("--output_shp",
                        type=str,
                        help="Output path to store shapefile",
                        default="footprint.shp")
    parser.add_argument("--out_crs",
                        type=int,
                        default=3031,
                        help="ESPG 4 digit number for output shapefile CRS")

    parser.add_argument(
        "--cores",
        nargs='?',
        type=int,
        help="Number of cores to use in parallel processing. Defaults to: (available cores - 1)")
    parser.add_argument(
        "--patch_size",
        type=int,
        default=500,
        help=
        "Patch-size for sea ice extraction. Patch edges are checked for Polygons spanning multiple patches."
    )
    parser.add_argument(
        "--polygon",
        type=int,
        default=0,
        help=
        "Boolean for whether the output shapefile will contain Polygons (large shapefile) or Polygon centroids + area (smaller file size)"
    )
    return parser.parse_args()


# Patch class for sea ice extraction
class Patch:
    def __init__(self, src_raster, col_off, row_off, patch_size=300):
        self.src_raster = src_raster
        self.window = Window(col_off=col_off, row_off=row_off, width=patch_size, height=patch_size)
        self.row = int(row_off / patch_size)
        self.col = int(col_off / patch_size)
        self.crs = None
        self.mask = None
        self.transforms = None
        self.polygons = False
        self.is_border = False

    def extract_mask(self):
        with rasterio.open(self.src_raster) as src:
            self.mask = src.read(1, window=self.window)
            self.transforms = src.window_transform(self.window)
            self.crs = src.crs

    def extract_ice_pol(self):
        self.polygons, self.is_border = polygonize_raster(extract_sea_ice(self.mask),
                                                          self.transforms)


class ShapefileWriter:
    def __init__(self, out_shp, crs, sizelimit=None):
        self.max_size = sizelimit or int(2e9 - 1e8)

        self.crs = crs
        self.schema = {'geometry': 'Point', 'properties': {'area': 'float', 'scene': 'str:150'}}

        self.name = [out_shp]
        self.base_name = os.path.basename(self.name[-1])
        self.current_file()
        print("Writing output to ", self.name[-1])

    def make_empty(self):
        print(self.name)
        if not os.path.isfile(self.name[-1]):
            print("Making shapefile ({})".format(self.name[-1]))
            sf = fiona.open(self.name[-1], 'w', 'ESRI Shapefile', self.schema, crs=self.crs)
            sf.close()

    def current_file(self, force_new=False):
        # if current name doesnt exist, create file
        if not os.path.isfile(self.name[-1]):
            self.make_empty()
            return self.name[-1]
        statinfo = os.stat(self.name[-1])
        if statinfo.st_size > self.max_size or force_new:
            self.name.append("{}({}).shp".format(self.base_name, len(self.name)))
            self.make_empty()
        return self.name[-1]

    def write_patch(self, patch):
        file = self.current_file()
        with fiona.open(file, 'a') as layer:
            geometry = patch.geom
            if self.crs != patch.crs:
                geometry = transform(patch.crs, self.crs, *(geometry.coords[0]))
            layer.write({
                'geometry': mapping(geometry),
                'properties': {
                    'area': patch.area,
                    'scene': patch.scene
                }
            })


def main():
    # defined patch tuple
    Floe = namedtuple('Floe', ('crs', 'geom', 'area', 'scene'))

    # read arguments
    args = parse_args()
    patch_size = args.patch_size

    # find input scenes
    input_scenes = []
    for path, _, filenames in os.walk(args.input_dir):
        for fname in filenames:
            if fname.endswith('.tif'):
                input_scenes.append(f"{path}/{fname}")

    # write output folder
    os.makedirs(os.path.dirname(args.output_shp), exist_ok=True)

    # extract sea ice
    stride = 0.99
    print(f"Extracting sea ice from {len(input_scenes)} scenes:")
    tic = time.time()

    writer = ShapefileWriter(args.output_shp, from_epsg(args.out_crs))

    def process_scene(scene: str):
        """Helper function to extract sea ice from rasters. Thresholds image to extract 
        lighter objects in a darker background. Trehsolded image is converted to a list 
        of shapely polygons, one for each object.
        
        Arguments:
            scene {str} -- path to input raster
        
        Returns:
            gpd.GeoDataFrame -- GeoPandas dataframe with polygons or polygon centroids 
            + area for sea ice patches.  
        """
        # read input scene and tile into patches
        pols = []
        with rasterio.open(scene) as scn_pan:
            nrows, ncols = scn_pan.shape
            src_crs = scn_pan.crs
            offsets = product(range(0, nrows, int(patch_size * stride)),
                              range(0, ncols, int(patch_size * stride)))
        for row, col in offsets:
            # read window
            curr = Patch(src_raster=scene, row_off=row, col_off=col, patch_size=patch_size)

            # extract sea ice polygon
            curr.extract_mask()
            curr.extract_ice_pol()

            # write center polygons and save border polygons to merge
            if curr.polygons:
                for idx, state in enumerate(curr.is_border):
                    pol = curr.polygons[idx]
                    if state:
                        pols.append(pol)
                    else:
                        writer.write_patch(
                            Floe(curr.crs, pol.centroid, pol.area, os.path.basename(scene)))

        # merge polygons that span multiple patches and return dataframe
        sea_ice = gpd.GeoDataFrame(geometry=pols,
                                   crs=src_crs,
                                   index=[ele for ele in range(len(pols))])
        sea_ice = sea_ice.assign(geometry=sea_ice.buffer(0))

        overlap_matrix = sea_ice.apply(lambda x: sea_ice.overlaps(x)).values.astype(int)
        n, ids = connected_components(overlap_matrix)
        sea_ice = gpd.GeoDataFrame({'geometry': sea_ice, 'group': ids})
        sea_ice = sea_ice.dissolve(by='group')
        sea_ice = sea_ice.to_crs(from_epsg(args.out_crs))

        # return polygons or centroids
        if args.polygon:
            return sea_ice
        else:
            sea_ice = sea_ice.assign(area=[pol.area for pol in list(sea_ice.geometry)])
            sea_ice = sea_ice.assign(geometry=[pol.centroid for pol in list(sea_ice.geometry)])
            return sea_ice

    # create processing pool and process all input scenes in parallel
    if args.cores is not None:
        pool = Pool(args.cores)
    else:
        pool = Pool(cpu_count() - 1)

    results = pool.map(process_scene, input_scenes)
    for scene in input_scenes:

        results.append(process_scene(scene))

    # write output to shapefile
    out = gpd.GeoDataFrame(crs=args.crs)
    for scn in results:
        out = out.append(scn, ignore_index=True)

    toc = time.time()
    print(
        f"Finished processing {len(input_scenes)} scenes in {(toc - tic) // 60} minutes and {(toc - tic) % 60} seconds"
    )


if __name__ == "__main__":
    main()
