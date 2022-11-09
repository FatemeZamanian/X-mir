# encoding: utf-8

"""
Read images and corresponding labels.
"""

import os
import csv
import torch
from PIL import Image
from torch.utils.data import Dataset


class ISICDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, use_melanoma=True, mask_dir=None, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            use_melanoma: whether or not to use melanoma samples (default = True).
            mask_dir: optional path to segmentation masks directory.
            transform: optional transform to be applied on a sample.
        """
        image_names = []
        labels = []
        mask_names = []
        with open(image_list_file, newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for line in reader:
                image_name = line[0]+'.jpg'
                if float(line[1]) == 1:
                    label = 2  # melanoma
                elif float(line[2]) == 1:
                    label = 1  # seborrheic keratosis
                else:
                    label = 0  # nevia
                if label == 2 and use_melanoma is False:
                    continue
                if mask_dir is not None:
                    raise NotImplementedError
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.mask_names = mask_names
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new('RGB', image.size), mask)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)


class ChestXrayDataSet(Dataset):
    def __init__(self, data_dir, image_list_file, use_covid=True, mask_dir=None, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            use_covid: whether or not to use COVID-19 samples (default = True).
            mask_dir: optional path to segmentation masks directory.
            transform: optional transform to be applied on a sample.
        """
        mapping = {
            'positive': 0,
            'negative': 1,
            'COVID-19': 2
        }

        image_names = []
        labels = []
        mask_names = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[1]
                label = mapping[items[2]]
                if label == 1 and use_covid is False:
                    continue
                if mask_dir is not None:
                    mask_name = os.path.join(
                        mask_dir, os.path.splitext(image_name)[0] + '_xslor.png')
                    mask_names.append(mask_name)
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.mask_names = mask_names
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item

        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.mask_names:
            mask_name = self.mask_names[index]
            mask = Image.open(mask_name).resize(image.size)
            image = Image.composite(image, Image.new('RGB', image.size), mask)
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.image_names)
