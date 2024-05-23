import os
import sys
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
from skimage import io
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.path.append('/data/psotnicz/projects/ProtoMIL')
from settings import MITO_SETTINGS

patch_size = MITO_SETTINGS.img_size
overlay = int(MITO_SETTINGS.overlay * patch_size)

print(f'rows and columns of pixels lost = {(4096 - overlay) % (patch_size - overlay)}')

def normalize_image(image):
    clip = 0.02
    min_ = np.quantile(image, clip)
    max_ = np.quantile(image, 1.0 - clip)
    image = np.clip(image, min_, max_)

    mean_value = np.mean(image)

    scaling_factor = 256.0 / (np.max(image) - np.min(image))
    image = (image - mean_value) * scaling_factor + 128
    image = np.clip(image, 0.0, 255.0)
    return image.astype(np.uint8)

def extract_patches(img, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dim_x, dim_y = img.shape
    idx = 0

    img = normalize_image(img)
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

def process_image_file(dir, path):
    out_dir = dir.replace('mito_all', 'mito_all_patches_512')
    print(dir)
    img = io.imread(dir)
    extract_patches(img, out_dir)

if __name__ == '__main__':
    path = 'data/mito_all'
    dirs = get_all_tifs(path)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image_file, dir, path) for dir in dirs]
        for future in futures:
            future.result()  # This will re-raise any exceptions caught during file processing
