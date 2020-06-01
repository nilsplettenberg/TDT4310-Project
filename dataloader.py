from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils import data
import torch
from torch.nn import functional as F
from torch import nn
import typing
import numpy as np
import xml.etree.ElementTree as ET
import os
import re

from preprocess import embed_words, zero_pad
from utils import to_cuda

np.random.seed(0)


# class Dataset(data.Dataset):
#   'Characterizes a dataset for PyTorch'
#   def __init__(self, inputs, labels):
#         'Initialization'
#         self.labels = labels
#         self.input = inputs

#   def __len__(self):
#         'Denotes the total number of samples'
#         return len(self.input)

#   def __getitem__(self, index):
#         'Generates one sample of data'

#         # Load data and get label
#         X = self.input[index]
#         y = self.labels[index]

#         return X, y

def load_data(path = 'data/pan19-author-profiling-20200229/training/en/', num_classes=3):
    '''
    Reads tweets and truth from the given path. Returns a list of lists with tweets of a users 
    and a list of the gender of the corresponding user (labels)
    
    Parameters:
    path (str): path of the dataset

    Returns:
    (list, list): tuple of texts and genders
    '''
    datasets = []
    for file in os.listdir(path):
        if file.endswith(".xml"):
            datasets.append(file)

    tweets = {}
    for dataset in datasets:
        root = ET.parse(os.path.join(path,dataset)).getroot()
        tweet_texts = []
        # get text from tweets
        for type_tag in root.findall('documents/document'):
            text = type_tag.text
            tweet_texts.append(text)
        user_id = re.findall(r"(.*)\.xml", dataset)[0]
        tweets[user_id] = tweet_texts

    labels = {}
    # get truth
    with open(os.path.join(path, 'truth.txt')) as f:
        for line in f:
            user_id, _, gender = re.findall(r'([A-Za-z0-9]*):::(human|bot):::([a-z]*)', line)[0]
            labels[user_id] = gender

    # create lists for input and output
    x, y =([] for i in range(2))

    # torch needs integer as output class
    # if num_classes==3:
    #     class_lables = {"bot":0, "female":1, "male":2}
    # else:
    #     class_lables = {"bot":0, "female":1, "male":1}
    class_lables = {"bot":0, "female":1, "male":2}


    for key, value in tweets.items():
        x.append(value)
        y.append(class_lables[labels[key]])

    return x,y

def get_data_loader(datasets, batch_size: int, validation_fraction: float = 0.1, dimensions: int = 25
                 ) -> typing.List[torch.utils.data.DataLoader]:

    x, y= datasets

    # # convert to torch tensor according to model type
    if type(x[0][0]) == list:
        # lstm requires float vectors
        x = torch.tensor(x, dtype=torch.float)
    else:
        # embedding layer requires long 
        x = torch.tensor(x, dtype=torch.long)
    y = torch.tensor(y)

    # split into train and test set, 0.9/0.1
    test_ind = int(len(x)*0.9) 
    x_test, y_test = x[test_ind:],y[test_ind:]
    x_train,y_train = x[:test_ind], y[:test_ind]

    data_train = data.TensorDataset(x_train, y_train)
    data_test = data.TensorDataset(x_test, y_test)
    
    # split train into train and validation
    indices = list(range(len(x_test)))
    split_idx = int(np.floor(validation_fraction * len(x_test)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=1,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=1)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=1)
    # dataloader_test = None

    return dataloader_train, dataloader_val, dataloader_test
    