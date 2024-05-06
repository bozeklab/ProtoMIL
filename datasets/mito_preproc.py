import os
import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob

from skimage import io
from PIL import Image

patch_size = 256
overlay = int(0.0 * patch_size)

def extract_patches(img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dim_x, dim_y = img.shape
    idx = 0
    img = np.pad(img, [(0, 2*patch_size), (0, 2*patch_size)], mode='constant')

    for x in range(0, dim_x, patch_size - overlay):
        for y in range(0, dim_y, patch_size - overlay):
            patch = img[x:x + patch_size, y:y + patch_size]
            Image.fromarray(patch).save(os.path.join(out_dir, str(idx) + ".jpg"))
            print('patch', idx)
            idx += 1


def get_all_tifs(path):
    dirs = glob.glob(path + '/**/*.tif', recursive=True)
    dirs.sort()
    return dirs

if __name__ == '__main__':
    path = 'data/mito_test'
    for dir in get_all_tifs(path):
        out_dir = dir.replace('mito_test', 'mito_test_patches')
        print(dir)
        img = io.imread(dir)
        extract_patches(img, out_dir)
        print('done', path)
