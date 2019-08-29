# Copyright (c) 2019 Bento Goncalves
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

__all__ = ['extract_sea_ice']

import cv2
import numpy as np

def extract_sea_ice(img, outline=False, kernel_size=9):
    if type(img) == str:
        img = cv2.imread(img, 0)

    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    if outline:
        img_blur = cv2.medianBlur(img, 5)
        img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 11, 2)
    else:
        kernel = np.ones([kernel_size, kernel_size])
        img_blur = cv2.medianBlur(img, 5)
        _, img_thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        img_thresh = cv2.erode(img_thresh, kernel)
        img_thresh = cv2.dilate(img_thresh, kernel)

    return img_thresh