import csv
import os.path

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor
import glob


class RCCPreprocessedBagsCross(data_utils.Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=3, all_labels=False, max_bag=20000,
                 positive_class_name='KIRP'):
        self.path = path
        self.train = train
        self.test = test
        self.folds = folds
        self.fold_id = fold_id
        self.random_state = random_state
        self.push = push
        self.all_labels = all_labels
        self.shuffle_bag = shuffle_bag
        self.data_augmentation = data_augmentation
        self.location_info = loc_info
        self.r = np.random.RandomState(random_state)
        self.max_bag = max_bag
        self.labels = {}
        self.positive_class_name = positive_class_name

        self.dir_list = self.get_dir_list(self.path)
        self.dir_labels = [1 if positive_class_name in d else 0 for d in self.dir_list]

        X_trainval, X_test, Y_trainval, Y_test = train_test_split(self.dir_list, self.dir_labels, test_size=0.25,
                                                                  random_state=fold_id,
                                                                  shuffle=True, stratify=self.dir_labels)
        X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size=0.2,
                                                          random_state=fold_id,
                                                          shuffle=True, stratify=Y_trainval)
        if train:
            self.dir_list = X_train
            self.dir_labels = Y_train
        elif test:
            self.dir_list = X_test
            self.dir_labels = Y_test
        else:
            self.dir_list = X_val
            self.dir_labels = Y_val
        self.embed_name = 'embeddings.pth'

    @staticmethod
    def get_dir_list(path):
        dirs = glob.glob(os.path.join(path, '*.tif'))
        dirs.sort()
        return dirs

    @classmethod
    def load_raw_image(cls, path):
        return to_tensor(pil_loader(path))

    class LazyLoader:
        def __init__(self, path, dir, indices):
            self.path = path
            self.dir = dir
            self.indices = indices

        def __getitem__(self, item):
            return RCCPreprocessedBagsCross.load_raw_image(
                os.path.join(self.dir, 'patch.{}.jpg'.format(int(self.indices[item]))))

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        dir = self.dir_list[index]
        label = self.dir_labels[index]
        try:
            bag = torch.load(os.path.join(dir, self.embed_name))
        except:
            print(dir)
            raise

        if bag.shape[0] > self.max_bag:
            if self.train:
                rng = np.random
            else:
                rng = np.random.default_rng(3)
            indices = rng.permutation(bag.shape[0])[:self.max_bag]
            bag = bag[indices].detach().clone()
        else:
            indices = np.arange(0, bag.shape[0])
        if self.all_labels:
            label = torch.LongTensor([label] * bag.shape[0])
        else:
            label = torch.LongTensor([label]).max().unsqueeze(0)

        if self.push:
            return self.LazyLoader(self.path, dir, indices), bag, label
        else:
            return bag, label


if __name__ == '__main__':
    ds = RCCPreprocessedBagsCross(path="../data/RCC_patches", test=True, train=False, all_labels=True, fold_id=1,
                                  folds=10, random_state=3, push=False)
    print(len(ds))
