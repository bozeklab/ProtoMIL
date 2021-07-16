import csv
import os.path

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import StratifiedKFold
from torchvision.datasets.folder import pil_loader
from torchvision.transforms.functional import to_tensor


class CamelyonPreprocessedBagsCross(data_utils.Dataset):
    def __init__(self, path, train=True, test=False, push=False, shuffle_bag=False, data_augmentation=False,
                 loc_info=False, folds=10, fold_id=1, random_state=3, all_labels=False, max_bag=20000):
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
        with open(os.path.join(path, 'labels.csv')) as f:
            reader = csv.DictReader(f, delimiter=',')
            for row in reader:
                self.labels[row['slide']] = 0 if row['label'] == 'normal' else 1
        self.embed_name = 'embeddings.pth'
        self.dir_list = [d for d in self.labels.keys() if os.path.exists(os.path.join(path, d, self.embed_name))]
        if self.train:
            self.dir_list = [d for d in self.dir_list if 'test' not in d]
        else:
            self.dir_list = [d for d in self.dir_list if 'test' in d]

    class LazyLoader:
        def __init__(self, path, dir):
            self.path = path
            self.dir = dir

        def __getitem__(self, item):
            return to_tensor(pil_loader(os.path.join(self.path, self.dir, 'patch.{}.jpg'.format(item))))

    def __len__(self):
        return len(self.dir_list)

    def __getitem__(self, index):
        dir = self.dir_list[index]
        try:
            bag = torch.load(os.path.join(self.path, dir, self.embed_name))
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
            pass
        if self.all_labels:
            label = torch.LongTensor([self.labels[dir]] * bag.shape[0])
        else:
            label = torch.LongTensor([self.labels[dir]]).max().unsqueeze(0)

        if self.push:
            return self.LazyLoader(self.path, dir), bag, label
        else:
            return bag, label


if __name__ == '__main__':
    ds = CamelyonPreprocessedBagsCross(path="../data/CAMELYON_patches", train=False, all_labels=True, fold_id=1,
                                       folds=10, random_state=3, push=False)
    print(len(ds))
