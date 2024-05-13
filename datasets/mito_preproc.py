import os
import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob

from skimage import io
from PIL import Image

sys.path.append('/data/psotnicz/projects/ProtoMIL')
from settings import MITO_SETTINGS

patch_size = MITO_SETTINGS.img_size
overlay = int(MITO_SETTINGS.overlay * patch_size)

print(f'rows and columns of pixels lost = {(4096 - overlay) % (patch_size - overlay)}')



def normalize_image(image):
    clip = 0.02
    min_ = np.quantile(img, clip)
    max_ = np.quantile(img, 1.0 - clip)
    np.clip(img, min_, max_)

    mean_value = np.mean(image)

    scaling_factor = 256.0 / (np.max(image) - np.min(image))
    normalized_image = (image - mean_value) * scaling_factor + 128
    return normalized_image.astype(np.uint8)


def extract_patches(img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dim_x, dim_y = img.shape
    idx = 0

    img = normalize_image(img)
    #img = np.pad(img, [(0, patch_size), (0, patch_size)], mode='constant')
    img_with_rects = img.copy()

    stride = patch_size - overlay
    for x in range(0, dim_x - patch_size + 1, stride):
        for y in range(0, dim_y - patch_size, stride):
            patch = img[x:x + patch_size, y:y + patch_size]
            Image.fromarray(patch).save(os.path.join(out_dir, str(idx) + ".jpg"))

            cv.rectangle(img_with_rects, (y, x), (y + patch_size, x + patch_size), 0, 5)
            idx += 1

    cv.imwrite(os.path.join(out_dir, 'patch_vis' + ".png"), img_with_rects)


def get_all_tifs(path):
    dirs = glob.glob(path + '/**/*.tif', recursive=True)
    dirs.sort()
    return dirs


if __name__ == '__main__':
    path = 'data/mito'
    for dir in get_all_tifs(path):
        out_dir = dir.replace('mito', 'mito_patches_512')
        print(dir)
        img = io.imread(dir)
        extract_patches(img, out_dir)
