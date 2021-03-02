"""Pytorch Dataset object that loads perfectly balanced MNIST dataset in bag form."""

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms


class MnistBags(data_utils.Dataset):
    def __init__(self, target_number=9, mean_bag_length=50, var_bag_length=1, num_bag=1000, seed=7, train=True,
                 push=False):
        self.target_number = target_number
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.seed = seed
        self.train = train
        self.push = push

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        self.pil_to_rgb_tensor = transforms.Compose([
            transforms.Lambda(lambda x: x.convert('RGB')),
            transforms.ToTensor(),
        ])

        self.normalize_to_tensor_transform = transforms.Compose([
            # transforms.Normalize((0.1307,), (0.3081,)),
        ])

        if self.train:
            self.train_bags_list, self.train_labels_list = self._form_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._form_bags()

    def _form_bags(self):
        if self.train:
            loader = data_utils.DataLoader(
                datasets.MNIST('data', train=True, download=True, transform=self.pil_to_rgb_tensor),
                batch_size=self.num_in_train,
                shuffle=False)
        else:
            loader = data_utils.DataLoader(
                datasets.MNIST('data', train=False, download=True, transform=self.pil_to_rgb_tensor),
                batch_size=self.num_in_test,
                shuffle=False)

        bags_list = []
        labels_list = []
        valid_bags_counter = 0
        label_of_last_bag = 0
        numbers, labels = next(iter(loader))

        num_in_case = self.num_in_train if self.train else self.num_in_test

        while valid_bags_counter < self.num_bag:
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            indices = torch.LongTensor(self.r.randint(0, num_in_case, bag_length))
            labels_in_bag = labels[indices]

            if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                labels_in_bag = labels_in_bag >= self.target_number
                labels_list.append(labels_in_bag.to(torch.long))
                bags_list.append(numbers[indices])
                label_of_last_bag = 1
                valid_bags_counter += 1
            elif label_of_last_bag == 1:
                index_list = []
                bag_length_counter = 0
                while bag_length_counter < bag_length:
                    index = torch.LongTensor(self.r.randint(0, num_in_case, 1))
                    label_temp = labels[index]
                    if label_temp.numpy()[0] != self.target_number:
                        index_list.append(index)
                        bag_length_counter += 1

                index_list = torch.cat(index_list)
                labels_in_bag = labels[index_list]
                labels_in_bag = labels_in_bag >= self.target_number
                labels_list.append(labels_in_bag.to(torch.long))
                bags_list.append(numbers[index_list])
                label_of_last_bag = 0
                valid_bags_counter += 1
            else:
                pass

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = torch.LongTensor([max(self.train_labels_list[index])])
        else:
            bag = self.test_bags_list[index]
            label = torch.LongTensor([max(self.test_labels_list[index])])

        if self.push:
            return bag, self.normalize_to_tensor_transform(bag), label
        else:
            return self.normalize_to_tensor_transform(bag), label
