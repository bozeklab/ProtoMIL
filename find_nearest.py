import heapq
import os
import time
from functools import cached_property

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from helpers import makedir, find_high_activation_crop
from receptive_field import compute_rf_prototype


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)


class ImagePatch:

    def __init__(self, patch, label, distance,
                 original_img=None, act_pattern=None, patch_indices=None):
        self.label = label
        self.negative_distance = -distance

        self.patch = patch
        self.original_img = original_img
        self.act_pattern = act_pattern
        self.patch_indices = patch_indices

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class ImagePatchLazy:

    def __init__(self, label, distance, batch_idx, image_idx, dataset, protoL_rf_info, distance_map_j,
                 prototype_activation_function_in_numpy, prototype_activation_function, max_dist, epsilon):
        self.label = label
        self.negative_distance = -distance
        self.batch_idx = batch_idx
        self.image_idx = image_idx
        self.dataset = dataset
        self.protoL_rf_info = protoL_rf_info
        self.distance_map_j = distance_map_j
        self.prototype_activation_function_in_numpy = prototype_activation_function_in_numpy
        self.prototype_activation_function = prototype_activation_function
        self.max_dist = max_dist
        self.epsilon = epsilon

    @cached_property
    def original_raw(self):
        return self.dataset[self.batch_idx][0][self.image_idx]

    @cached_property
    def closest_patch_indices_in_img(self):
        closest_patch_indices_in_distance_map_j = \
            list(np.unravel_index(np.argmin(self.distance_map_j, axis=None),
                                  self.distance_map_j.shape))
        closest_patch_indices_in_distance_map_j = [0] + closest_patch_indices_in_distance_map_j
        closest_patch_indices_in_img = \
            compute_rf_prototype(self.original_raw.size(1),
                                 closest_patch_indices_in_distance_map_j,
                                 self.protoL_rf_info)
        return closest_patch_indices_in_img

    @cached_property
    def patch(self):
        closest_patch_indices_in_img = self.closest_patch_indices_in_img
        closest_patch = \
            self.original_raw[:,
            closest_patch_indices_in_img[1]:closest_patch_indices_in_img[2],
            closest_patch_indices_in_img[3]:closest_patch_indices_in_img[4]]
        closest_patch = closest_patch.numpy()
        closest_patch = np.transpose(closest_patch, (1, 2, 0))
        return closest_patch

    @cached_property
    def original_img(self):
        original_img = self.original_raw.numpy()
        original_img = np.transpose(original_img, (1, 2, 0))
        return original_img

    @cached_property
    def act_pattern(self):
        if self.prototype_activation_function == 'log':
            act_pattern = np.log(
                (self.distance_map_j + 1) / (self.distance_map_j + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            act_pattern = self.max_dist - self.distance_map_j
        else:
            act_pattern = self.prototype_activation_function_in_numpy(self.distance_map_j)
        return act_pattern

    @cached_property
    def patch_indices(self):
        patch_indices = self.closest_patch_indices_in_img[1:5]
        return patch_indices

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


class ImagePatchInfo:

    def __init__(self, label, distance):
        self.label = label
        self.negative_distance = -distance

    def __lt__(self, other):
        return self.negative_distance < other.negative_distance


# find the nearest patches in the dataset to each prototype
def find_k_nearest_patches_to_prototypes(dataloader,  # pytorch dataloader (must be unnormalized in [0,1])
                                         ppnet,  # pytorch network with prototype_vectors
                                         k=5,
                                         preprocess_input_function=None,  # normalize if needed
                                         full_save=False,  # save all the images
                                         root_dir_for_saving_images='./nearest',
                                         log=print,
                                         prototype_activation_function_in_numpy=None,
                                         only_n_most_activated=None):
    ppnet.eval()
    '''
    full_save=False will only return the class identity of the closest
    patches, but it will not save anything.
    '''
    print('        find nearest patches')
    start = time.time()
    n_prototypes = ppnet.num_prototypes

    prototype_shape = ppnet.prototype_shape
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    protoL_rf_info = ppnet.proto_layer_rf_info

    heaps = []
    # allocate an array of n_prototypes number of heaps
    for _ in range(n_prototypes):
        # a heap in python is just a maintained list
        heaps.append([])

    with tqdm(total=len(dataloader.dataset), unit='bag') as pbar:
        for idx, (search_batch_raw, search_batch, search_y) in enumerate(dataloader):
            with torch.no_grad():
                search_batch = search_batch.cuda()
                ppnet.forward(search_batch)
                proto_dist_torch = ppnet.distances
                attention = ppnet.A
            raw_sample = search_batch_raw[0]

            # protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
            proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

            if only_n_most_activated:
                most_activates = list(
                    torch.topk(attention, min(only_n_most_activated, attention.shape[1]), dim=-1, largest=True)[1].detach().cpu().numpy()[0])

            for img_idx, distance_map in enumerate(proto_dist_):
                if only_n_most_activated:
                    if img_idx not in most_activates:
                        continue

                for j in range(n_prototypes):
                    # find the closest patches in this batch to prototype j

                    closest_patch_distance_to_prototype_j = np.amin(distance_map[j])

                    if full_save:
                        closest_patch = ImagePatchLazy(
                            label=search_y[0],
                            distance=closest_patch_distance_to_prototype_j,
                            batch_idx=idx,
                            image_idx=img_idx,
                            dataset=dataloader.dataset,
                            protoL_rf_info=protoL_rf_info,
                            distance_map_j=distance_map[j],
                            prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
                            prototype_activation_function=ppnet.prototype_activation_function,
                            max_dist=max_dist,
                            epsilon=ppnet.epsilon
                        )
                    else:
                        closest_patch = ImagePatchInfo(label=search_y[0],
                                                       distance=closest_patch_distance_to_prototype_j)

                    # add to the j-th heap
                    if len(heaps[j]) < k:
                        heapq.heappush(heaps[j], closest_patch)
                    else:
                        # heappushpop runs more efficiently than heappush
                        # followed by heappop
                        heapq.heappushpop(heaps[j], closest_patch)
            pbar.update(1)

    # after looping through the dataset every heap will
    # have the k closest prototypes
    for j in range(n_prototypes):
        # finally sort the heap; the heap only contains the k closest
        # but they are not ranked yet
        heaps[j].sort()
        heaps[j] = heaps[j][::-1]

        if full_save:

            dir_for_saving_images = os.path.join(root_dir_for_saving_images,
                                                 str(j))
            makedir(dir_for_saving_images)

            labels = []

            for i, patch in enumerate(heaps[j]):
                # save the activation pattern of the original image where the patch comes from
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i + 1) + '_act.npy'),
                        patch.act_pattern)

                # save the original image where the patch comes from
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i + 1) + '_original.png'),
                           arr=patch.original_img,
                           vmin=0.0,
                           vmax=1.0)

                # overlay (upsampled) activation on original image and save the result
                img_size = patch.original_img.shape[0]
                upsampled_act_pattern = cv2.resize(patch.act_pattern,
                                                   dsize=(img_size, img_size),
                                                   interpolation=cv2.INTER_CUBIC)
                rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
                rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
                heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = heatmap[..., ::-1]
                overlayed_original_img = 0.5 * patch.original_img + 0.3 * heatmap
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i + 1) + '_original_with_heatmap.png'),
                           arr=overlayed_original_img,
                           vmin=0.0,
                           vmax=1.0)

                # if different from original image, save the patch (i.e. receptive field)
                if patch.patch.shape[0] != img_size or patch.patch.shape[1] != img_size:
                    np.save(os.path.join(dir_for_saving_images,
                                         'nearest-' + str(i + 1) + '_receptive_field_indices.npy'),
                            patch.patch_indices)
                    plt.imsave(fname=os.path.join(dir_for_saving_images,
                                                  'nearest-' + str(i + 1) + '_receptive_field.png'),
                               arr=patch.patch,
                               vmin=0.0,
                               vmax=1.0)
                    # save the receptive field patch with heatmap
                    overlayed_patch = overlayed_original_img[patch.patch_indices[0]:patch.patch_indices[1],
                                      patch.patch_indices[2]:patch.patch_indices[3], :]
                    plt.imsave(fname=os.path.join(dir_for_saving_images,
                                                  'nearest-' + str(i + 1) + '_receptive_field_with_heatmap.png'),
                               arr=overlayed_patch,
                               vmin=0.0,
                               vmax=1.0)

                # save the highly activated patch    
                high_act_patch_indices = find_high_activation_crop(upsampled_act_pattern)
                high_act_patch = patch.original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                 high_act_patch_indices[2]:high_act_patch_indices[3], :]
                np.save(os.path.join(dir_for_saving_images,
                                     'nearest-' + str(i + 1) + '_high_act_patch_indices.npy'),
                        high_act_patch_indices)
                plt.imsave(fname=os.path.join(dir_for_saving_images,
                                              'nearest-' + str(i + 1) + '_high_act_patch.png'),
                           arr=high_act_patch,
                           vmin=0.0,
                           vmax=1.0)
                # save the original image with bounding box showing high activation patch
                imsave_with_bbox(fname=os.path.join(dir_for_saving_images,
                                                    'nearest-' + str(i + 1) + '_high_act_patch_in_original_img.png'),
                                 img_rgb=patch.original_img,
                                 bbox_height_start=high_act_patch_indices[0],
                                 bbox_height_end=high_act_patch_indices[1],
                                 bbox_width_start=high_act_patch_indices[2],
                                 bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            labels = np.array([patch.label for patch in heaps[j]])
            np.save(os.path.join(dir_for_saving_images, 'class_id.npy'),
                    labels)

    labels_all_prototype = np.array([[patch.label for patch in heaps[j]] for j in range(n_prototypes)])

    if full_save:
        np.save(os.path.join(root_dir_for_saving_images, 'full_class_id.npy'),
                labels_all_prototype)

    end = time.time()
    log('        find nearest patches time: \t{0}'.format(end - start))

    return labels_all_prototype
