"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import KFold
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms

datasets.MNIST.resources = [                   
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        ]

class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, bag_length_mean=200, bag_length_std=150, bag_length_min=50, bag_length_max=600,
                 positive_samples_in_bag_ratio_mean=0.1, positive_samples_in_bag_ratio_std=0.01,
                 num_bags_train=300, num_bags_test=300, folds=10, fold_id=0, random_state=3, train=True, push=False,
                 test=False, all_labels=False):
        self.target_number = target_number
        self.mean_bag_length = bag_length_mean
        self.var_bag_length = bag_length_std
        self.num_bag = num_bags_train if train else num_bags_test
        self.train = train
        self.push = push
        self.test = test
        self.negative_bags = self.num_bag // 2
        self.min_bag_size = bag_length_min
        self.max_bag_size = bag_length_max
        self.folds = folds
        self.fold_id = fold_id
        self.random_state = random_state
        self.all_labels = all_labels

        self.target_numbers_in_pos_bag_mean = positive_samples_in_bag_ratio_mean
        self.target_numbers_in_pos_bag_std = positive_samples_in_bag_ratio_std

        self.r = np.random.RandomState(random_state)

        self.pil_to_rgb_tensor = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
        ])

        self.normalize_to_tensor_transform = transforms.Compose([
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.bags_list, self.labels_list = self._form_bags()

    def _transform_single(self, x):
        return self.pil_to_rgb_tensor(x)

    def _form_bags(self):
        mnist_train = datasets.MNIST('data', train=True, download=True)
        mnist_test = datasets.MNIST('data', train=False, download=True)
        mnist = ConcatDataset([mnist_train, mnist_test])
        folds = list(KFold(n_splits=self.folds, shuffle=True, random_state=self.random_state).split(mnist))

        if self.test:
            indices = set(folds[self.fold_id][1])
        else:
            if self.train:
                val_indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]))
                indices = set(folds[self.fold_id][0]) - set(val_indices)
            else:  # valid
                indices = self.r.choice(folds[self.fold_id][0], len(folds[self.fold_id][1]))

        negative_samples = torch.stack(
            [self._transform_single(d) for idx, (d, t) in enumerate(mnist) if
             idx in indices and t != self.target_number])
        positive_samples = torch.stack(
            [self._transform_single(d) for idx, (d, t) in enumerate(mnist) if
             idx in indices and t == self.target_number])

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            bag_length = min(max(bag_length, self.min_bag_size), self.max_bag_size)

            if i < self.negative_bags:
                indices = self.r.randint(0, len(negative_samples), bag_length)
                bags_list.append(negative_samples[indices])
                labels_list.append(torch.zeros(bag_length, dtype=torch.long))
            else:
                positive_ratio = float(
                    self.r.normal(self.target_numbers_in_pos_bag_mean, self.target_numbers_in_pos_bag_std, 1))
                positive_num = min(max(int(bag_length * positive_ratio), 1), bag_length - 1)
                negative_num = bag_length - positive_num
                pos_indices = self.r.randint(0, len(positive_samples), positive_num)
                neg_indices = self.r.randint(0, len(negative_samples), negative_num)
                bag = torch.cat([positive_samples[pos_indices], negative_samples[neg_indices]])
                labels = torch.cat(
                    [torch.ones(positive_num, dtype=torch.long), torch.zeros(negative_num, dtype=torch.long)])
                shuffle_indices = self.r.permutation(bag_length)
                bag = bag[shuffle_indices]
                labels = labels[shuffle_indices]
                bags_list.append(bag)
                labels_list.append(labels)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.labels_list)
        else:
            return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        if self.all_labels:
            label = self.labels_list[index]
        else:
            label = self.labels_list[index].max().unsqueeze(0)
        if self.push:
            return bag, self.normalize_to_tensor_transform(bag), label
        else:
            return self.normalize_to_tensor_transform(bag), label
