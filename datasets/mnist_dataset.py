"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

import numpy as np
import torch
import torch.utils.data as data_utils
from PIL import Image
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=200, var_bag_length=150, min_bag_size=50, max_bag_size=600,
                 train_num_bag=1000, test_num_bags=100, seed=7, train=True,
                 push=False):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = train_num_bag if train else test_num_bags
        self.seed = seed
        self.train = train
        self.push = push
        self.negative_bags = self.num_bag // 2
        self.min_bag_size = min_bag_size
        self.max_bag_size = max_bag_size

        self.target_numbers_in_pos_bag_mean = 0.3
        self.target_numbers_in_pos_bag_std = 0.25

        self.r = np.random.RandomState(seed)

        self.pil_to_rgb_tensor = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
        ])

        self.normalize_to_tensor_transform = transforms.Compose([
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])

        if self.train:
            self.bags_list, self.labels_list = self._form_bags()
        else:
            self.bags_list, self.labels_list = self._form_bags()

    def _transform_single(self, x):
        img = Image.fromarray(x.numpy(), mode='L')
        return self.pil_to_rgb_tensor(img)

    def _form_bags(self):
        if self.train:
            data = datasets.MNIST('data', train=True, download=True, transform=self.pil_to_rgb_tensor)
        else:
            data = datasets.MNIST('data', train=False, download=True, transform=self.pil_to_rgb_tensor)

        negative_samples = torch.stack(
            [self._transform_single(d) for d, t in zip(data.data, data.targets) if t != self.target_number])
        positive_samples = torch.stack(
            [self._transform_single(d) for d, t in zip(data.data, data.targets) if t == self.target_number])

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
        label = self.labels_list[index].max().unsqueeze(0)
        if self.push:
            return bag, self.normalize_to_tensor_transform(bag), label
        else:
            return self.normalize_to_tensor_transform(bag), label
