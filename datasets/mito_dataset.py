import os.path
import glob

import random
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets.folder import pil_loader
from torchvision.transforms import transforms
from torchvision.transforms.functional import to_tensor

def get_dir_list(path):
    dirs = glob.glob(path, recursive=True)
    dirs.sort()
    return dirs

class MitoPreprocessedBagsCross(data_utils.Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=3, all_labels=False, max_bag=20000):
        self.path = path
        self.train = train
        self.test = test
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
        for p in get_dir_list(path + '/**/*.tif/'):
            self.labels[p] = 0 if 'fl_fl' in p else 1

        self.embed_name = 'embeddings.pth'

        # shuffle order of files
        dir_shuffled = list(self.labels.keys())
        random.seed(self.random_state)
        random.shuffle(dir_shuffled)
        self.dir_list = np.array([d for d in dir_shuffled
                                  if os.path.exists(os.path.join(d, self.embed_name))])

        # train val test split
        proportions = {0: 0.7, 1: 0.15, 2: 0.15}
        elements_count = {key: int(len(self.dir_list) * value) for key, value in proportions.items()}
        result_list = []
        for key, count in elements_count.items():
            result_list.extend([key] * count)
        random.shuffle(result_list)
        result_list = np.array(result_list)
        if self.train:
            self.dir_list = self.dir_list[result_list == 0]
        elif self.test:
            self.dir_list = self.dir_list[result_list == 1]
        else:
            self.dir_list = self.dir_list[result_list == 2]

    @classmethod
    def load_raw_image(cls, path):
        return to_tensor(pil_loader(path))

    class LazyLoader:
        def __init__(self, path, dir, indices):
            self.path = path
            self.dir = dir
            self.indices = indices

        def __getitem__(self, item):
            return MitoPreprocessedBagsCross.load_raw_image(
                os.path.join(self.dir, '{}.jpg'.format(int(self.indices[item]))))

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        dir = self.dir_list[index]
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
            label = torch.LongTensor([self.labels[dir]] * bag.shape[0])
        else:
            label = torch.LongTensor([self.labels[dir]]).max().unsqueeze(0)

        if self.push:
            return self.LazyLoader(self.path, dir, indices), bag, label
        else:
            return bag, label
