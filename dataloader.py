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

def loadData(path = 'data/pan19-author-profiling-20200229/training/en/'):
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
    class_lables = {"bot":0, "female":1, "male":2}

    for key, value in tweets.items():
        x.append(value)
        y.append(class_lables[labels[key]])

    return x,y

def load_data(batch_size: int, validation_fraction: float = 0.1, input_size: int = 32, dimension: int = 3
                 ) -> typing.List[torch.utils.data.DataLoader]:

    x_train, y_train = loadData("data/pan19-author-profiling-20200229/training/en/")
    x_test, y_test = loadData("data/pan19-author-profiling-20200229/test/en/")

    # embedd words using glove matrix
    x_train = embed_words(x_train[:1000])
    x_test = embed_words(x_test[:1000])
    # remove too short and too long sequences, padd with zeros 
    x_train = zero_pad(x_train,y_train)
    x_test = zero_pad(x_test,y_test)

    # data_train = Dataset(x_train, y_train)
    # data_test = Dataset(x_test, y_test)
    # # convert to torch tensor
    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = F.one_hot(torch.tensor(y_train[:len(x_train)]))
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = F.one_hot(torch.tensor(y_test[:len(x_test)]))


    data_train = data.TensorDataset(x_train, y_train)
    data_test = data.TensorDataset(x_test, y_test)
    
    
    indices = list(range(len(x_train)))
    split_idx = int(np.floor(validation_fraction * len(x_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_test, dataloader_val
    