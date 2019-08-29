import rasterio
import os
import re
import time
import traceback
from fiona.crs import from_epsg
import fiona
from shapely.geometry.geo import Polygon, mapping
from collections import namedtuple
import numpy as np
import geopandas as gpd
import datetime
import glob
import argparse
import multiprocessing

# footprint template
Footprint = namedtuple('Footprint', ('crs', 'geom', 'filename', 'meta'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="input directory containing rasters")
    parser.add_argument("--output", type=str, help="Output path to store shapefile", default="footprint.shp")
    parser.add_argument("--out_crs", type=int, default=3031, help="ESPG 4 digit number for output shapefile CRS")
    parser.add_argument("--cores", nargs='?', type=int,
                        help="Number of cores to use in parallel processing. Defaults to: (available cores - 1)")
    parser.add_argument("--log", type=str, default='footprint_log.txt', help="Logging text file")
    return parser.parse_args()


def find_season(year, month):
    if 12 >= month >= 6:
        return year
    elif month <= 5:
        return year - 1
        

def find_day_of_season(year, month, day, season):
    from datetime import datetime
    delta = datetime(year, month, day) - datetime(season, 5, 31)
    return delta.days


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

class DigitalGlobeSchema:
    def __init__(self, tags, filename=None):
        self.file = filename
    
    
    def parse_tags(self):
        # get sensor or replace with UNK for unknown sensor
        sensor = re.search(r'[A-Z]{2}\d{2}', self.file).group(0)[:4]
        # get acquisition date or replace with 1st Jan 9999
        match = re.search(r'\d{2}[A-Z]{3}\d{8}', self.file).group(0)
        date = '20' + match[0:2] + "%02d" % (time.strptime(match[2:5], '%b').tm_mon) + match[5:]
        date = datetime.datetime.strptime(date, '%Y%m%d%H%M%S')
        date_str = str(date)
        season = find_season(date.year, date.month)
        seasonday = find_day_of_season(date.year, date.month, date.day, season)
        return {'sensor': sensor,
                'date': date_str,
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'season': season,
                'dayofseaso': seasonday
                }
    

class Tiff:
    def __init__(self, tiff):
        self.file = tiff
        self.crs = None
        self.transform = None
        self.bounds = None
        self.meta_data = {}
        self.extract_meta_data()
        self.meta_data.update({'location': tiff})
        self.meta_data.update({'id': os.path.split(self.file)[1]})
    def extract_meta_data(self):
        with rasterio.open(self.file) as input_raster:
            self.transform = input_raster.transform
            self.crs = input_raster.crs
            self.bounds = input_raster.bounds
            self.meta_data = DigitalGlobeSchema(input_raster.tags(), self.file).parse_tags()
    def get_trimmed_geom(self):
        return get_exact_trimmed_geom(self.file)


def extract_footprint(tiff):
    tif = Tiff(tiff)
    return Footprint(tif.crs, tif.get_trimmed_geom(), tif.file, tif.meta_data)


def extract_footprint_worker(tiff, result_queue):
    footprint = extract_footprint(tiff)
    result_queue.put(footprint)
    return tiff


class ShapefileWriter:
    def __init__(self, name, crs, sizelimit=None):
        self.max_size = sizelimit or int(2e9 - 1e8)
        self.crs = from_epsg(crs)
        self.schema = {'geometry': 'Polygon', 'properties': {'location': 'str:150',
                                                             'sensor': 'str:16',
                                                             'date': 'str:19',
                                                             'year': 'int',
                                                             'month': 'int',
                                                             'day': 'int',
                                                             'season': 'int',
                                                             'dayofseaso': 'int',
                                                             'id': 'str:150'}}
        self.base_name = os.path.splitext(name)[0]
        self.name = [name]
        self.current_file()
        print(f"Writing output to {self.name[-1]}")

    def make_empty(self):
        if not os.path.isfile(self.name[-1]):
            print(f"Making shapefile ({self.name[-1]})")
            sf = fiona.open(self.name[-1], 'w', 'ESRI Shapefile', self.schema, crs=self.crs)
            sf.close()

    def current_file(self):
        if not os.path.isfile(self.name[-1]):
            self.make_empty()
            return self.name[-1]
        statinfo = os.stat(self.name[-1])
        if statinfo.st_size > self.max_size:
            self.name.append(f"{self.base_name}({len(self.name)}).shp")
            self.make_empty()
        return self.name[-1]

    def write_footprint(self, footprint):
        file = self.current_file()
        print(f"Writing {footprint.filename} to {file}")
        print(footprint.meta)
        with fiona.open(file, 'a') as layer:
            layer.write({'geometry': footprint.geom,
                         'properties': footprint.meta})


def write_footprint(shapefile, crs, result_queue, log_file):
    sf = ShapefileWriter(shapefile, crs=crs)
    while True:
        try:
            footprint = result_queue.get()

            # if failed
            if footprint.geom is None:
                with open(log_file, 'a') as log:
                    print(f"Error processing {footprint.filename}... {footprint.crs}", file=log)
                continue
            
            # if last item on queue
            if footprint.geom == "kill":
                with open(log_file, 'a') as log:
                    print("Rasters complete. Closing shapefile writer.", file=log)
                return 0

            # if succeeded
            sf.write_footprint(footprint)
            with open(log_file, 'a') as log:
                print(f"Wrote {footprint.filename} footprint", file=log)

        except:
            with open(log_file, 'a') as log:
                print(f"Error while writing {footprint.filename}", file=log)
            traceback.print_exc()


def process_footprints(files):
    aresults = []
    failed = []
    for file in files:
        try:
            results = extract_footprint(file)
            aresults.append(results)
        except:
            print(failed)
    return aresults


def bulk_process_footprints(files, output, crs, cores=1, log_file=None):
    log_file = log_file or "footprint_processing_log.txt"
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    n_proc = max(2, cores or multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(n_proc)

    # deploy listener
    watcher = pool.apply_async(write_footprint, (output, crs, result_queue, log_file))
    jobs = []

    for file in files:
        job = pool.apply_async(extract_footprint_worker, (file, result_queue))
        jobs.append(job)

    results = []
    failed = []

    for job, file in zip(jobs, files):
        try:
            result = job.get(360), True
            results.append(result)
        except:
            with open(log_file, 'a') as log:
                print(f"{file} timed out", file=log)
            failed.append(file)

    result_queue.put(Footprint(None, "kill", None, None))
    pool.close()
    watcher.get()

    return {'succeeded': results, "failed": failed}


if __name__ == "__main__":
    args = parse_args()
    output = args.output
    output_dir, output_shapefile = os.path.split(args.output) 
    log_file = os.path.join(output_dir, args.log)

    os.makedirs(output_dir, exist_ok=True)

    files = []
    try:
        existing = list(gpd.read_file(args.output).id)
        for (dirpath, dirnames, filenames) in os.walk(args.input):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.tif') and file not in existing]
    except: 
        for (dirpath, dirnames, filenames) in os.walk(args.input):
            files += [os.path.join(dirpath, file) for file in filenames if file.endswith('.tif')]

    

    start_time = datetime.datetime.now()
    with open(log_file, 'w') as log:
        print(f"Processing {len(files)} rasters @ {start_time}", file=log)

    results = bulk_process_footprints(files, output, crs=args.out_crs, cores=args.cores, log_file=log_file)
    print(results)

    with open(log_file, 'a') as log:
        print(f"Finished processing {len(results['succeeded'])} (of {len(files)}) rasters in {datetime.datetime.now() - start_time}", file=log)
