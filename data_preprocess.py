import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, RandomSampler, SequentialSampler
import torchvision.transforms as transforms
from PIL import Image
import pickle as pkl
import json
import sys

class RetinopathyDataset(Dataset):
    """Diabetic Retinopathy dataset from https://www.kaggle.com/c/diabetic-retinopathy-detection/data."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file filename information.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data_frame.iloc[idx, 0] + '.jpeg')
        
        image = Image.open(img_name)
                    
        if self.transform:
            image = self.transform(image)

        image_class = self.data_frame.iloc[idx, -1]
        sample = {'img': image, 'label': image_class}

        return sample


if __name__ == '__main__':
    img_size = int(sys.argv[1])
    image_path = os.getcwd()

    transformations = transforms.Compose([
        transforms.Resize(img_size + 10),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    dataset = RetinopathyDataset(csv_file='trainLabels.csv',
                                 root_dir=os.path.join(image_path, 'images'), 
                                 transform = transformations)

    # split dataset into train, valid, test by 0.6, 0.2, 0.2
    train_len = int(0.6 * dataset.__len__())
    valid_len = int(0.2 * dataset.__len__())
    test_len = dataset.__len__() - train_len - valid_len

    train_dataset, test_dataset, valid_dataset = random_split(dataset, [train_len, test_len, valid_len])

    # Set batch size for the DataLoader
    batch_size = 32

    # Create the DataLoaders for our training and validation sets.
    # We'll take training samples in random order. 
    train_dataloader = DataLoader(
                train_dataset,  # The train samples.
                sampler = RandomSampler(train_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For test the order doesn't matter, so we'll just read them sequentially.
    test_dataloader = DataLoader(
                test_dataset,  # The test samples.
                sampler = SequentialSampler(test_dataset), # Select batches randomly
                batch_size = batch_size # Trains with this batch size.
            )

    # For validation the order doesn't matter, so we'll just read them sequentially.
    valid_dataloader = DataLoader(
                valid_dataset, # The validation samples.
                sampler = SequentialSampler(valid_dataset), # Pull out batches sequentially.
                batch_size = batch_size # Evaluate with this batch size.
            )

    print('train data size: ', len(train_dataloader))
    print('test data size: ', len(test_dataloader))
    print('valid data size: ', len(valid_dataloader))

    # save dataloader
    pkl.dump(train_dataloader, open("train_loader_{}.pkl".format(img_size), "wb"))
    pkl.dump(test_dataloader, open("test_loader_{}.pkl".format(img_size), "wb"))
    pkl.dump(valid_dataloader, open("valid_loader_{}.pkl".format(img_size), "wb"))

