import os
import sys

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from openslide import OpenSlide

LAYER = 1
MASK_LAYER = 7

PATCH_SIZE = 224

O_P_R = (2 ** LAYER)
O_M_R = (2 ** MASK_LAYER)

ORIG_PATCH_SIZE = PATCH_SIZE * O_P_R
MASK_PATCH_SIZE = ORIG_PATCH_SIZE // O_M_R

NON_EMPTY_RATIO = 0.3


def get_whole_slide(slide, level):
    return slide.read_region((0, 0), level, slide.level_dimensions[level])


def get_mask(slide, layer):
    img = np.array(get_whole_slide(slide, layer).convert('RGB'))
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img[gray < 80] = 255
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    lower_red = np.array([20, 20, 20])
    upper_red = np.array([200, 200, 200])
    mask = cv.inRange(hsv, lower_red, upper_red)
    close_kernel = np.ones((5, 5), dtype=np.uint8)
    image_close = cv.morphologyEx(np.array(mask), cv.MORPH_CLOSE, close_kernel)
    open_kernel = np.ones((11, 11), dtype=np.uint8)
    opening = cv.morphologyEx(np.array(image_close), cv.MORPH_OPEN, open_kernel)

    return opening


def extract_patches(slide, mask, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dim_x, dim_y = slide.level_dimensions[0]
    idx = 0
    for x in range(0, dim_x - ORIG_PATCH_SIZE, ORIG_PATCH_SIZE):
        for y in range(0, dim_y - ORIG_PATCH_SIZE, ORIG_PATCH_SIZE):
            mask_x, mask_y = x // O_M_R, y // O_M_R
            mask_sum = mask[mask_y:mask_y + MASK_PATCH_SIZE, mask_x:mask_x + MASK_PATCH_SIZE].sum()
            if mask_sum < NON_EMPTY_RATIO * MASK_PATCH_SIZE * MASK_PATCH_SIZE:
                continue
            patch = slide.read_region((x, y), LAYER, (PATCH_SIZE, PATCH_SIZE)).convert('RGB')
            patch.save(os.path.join(out_dir, 'patch.{}.jpg'.format(idx)), quality=90)
            print('patch', idx)
            idx += 1


if __name__ == '__main__':
    path = sys.argv[1]
    out_dir = path.replace('CAMELYON', 'CAMELYON_patches')
    with OpenSlide(path) as slide:
        print(slide.level_dimensions)
        im = get_whole_slide(slide, MASK_LAYER).convert('RGB')
        mask = get_mask(slide, MASK_LAYER)
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(im)
        ax[1].imshow(mask)
        fig.tight_layout()
        # plt.show()
        plt.savefig(out_dir + '.jpg')
        extract_patches(slide, mask, out_dir)
        print('done', path)
