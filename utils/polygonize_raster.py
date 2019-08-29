# Copyright (c) 2019 Bento Goncalves
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__all__ = ['polygonize_raster']

import cv2
from shapely.geometry import Polygon, LineString, Point


def polygonize_raster(mask, transforms, min_area=15):
    # write mask to polygon
    polygons = []
    border = LineString([
        Point(transforms * (0, 0)),
        Point(transforms * (0, mask.shape[1])),
        Point(transforms * (mask.shape[0], mask.shape[1])),
        Point(transforms * (mask.shape[0], 0)),
        Point(transforms * (0, 0))
    ])
    border = border.buffer(1)
    is_border = []
    edges = cv2.findContours(image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    for edge in edges:
        pol = Polygon([transforms * ele[0] for ele in edge])
        if pol.area > min_area:
            polygons.append(pol)
            is_border.append(pol.intersects(border))

    if len(polygons) > 0:
        return polygons, is_border
    else:
        return False, False
