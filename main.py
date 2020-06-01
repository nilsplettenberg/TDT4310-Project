import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import pickle

from dataloader import load_data, get_data_loader
from model import Glove_model, Embedding_Model
from trainer import Trainer, create_plots
from preprocess import preprocess, prepare_sequence, zero_pad

def map_binary(x):
    if x==2:
        return 1
    else: 
        return x

def remove_bot(x,y):
    new_x =[]
    new_y =[]
    for idx, label in enumerate(y):
        if label !=0:
            new_x.append(x[idx])
            # output of network will be 0 and 1
            new_y.append(label-1)
    return new_x, new_y
if __name__ == "__main__":
    epochs = 100
    batch_size = 16 # was 16
    learning_rate = 1e-3
    early_stop_count = 5
    dimensions = 200
    num_classes = 3
    detect_gender = False # if false and 2 classes: detect bot/human
    lang = "en" # es or en
    use_glove = True

    # load datasets
    x_train, y_train = load_data("data/pan19-author-profiling-20200229/training/"+ lang +"/", num_classes)
    x_test, y_test = load_data("data/pan19-author-profiling-20200229/test/"+ lang +"/", num_classes)

    # concatenate sets for preprocessing
    x = x_train + x_test
    y = y_train + y_test

    # # relable dataset for binary detection
    if detect_gender:
        if num_classes == 2:
            x,y = remove_bot(x,y)
    else:
        if num_classes == 2:
            # mapping 2 to 1 to have a binary output for only 2 classes
            y = list(map(map_binary, y))

    x, y, word_to_ix = preprocess(x, y, dimensions, lang=lang, glove=use_glove)

    datasets = (x,y)
    # # Saving the objects:
    # with open('/work/nilsple/data/preprocessed_nostop_gender'+ lang +'.pkl', 'wb') as f: 
    #     pickle.dump((datasets, word_to_ix), f)

    # # Getting preprocessed datasets
    # with open('/work/nilsple/data/preprocessed_nostop_gender'+ lang +'.pkl', 'rb') as f:
    #     datasets, word_to_ix = pickle.load(f)

    # x,y = datasets

    dataloaders = get_data_loader(datasets, batch_size, dimensions = dimensions)
    if use_glove:
        model = Glove_model(dimensions,num_classes, 200, 2, True)
    else:
        model = Embedding_Model(dimensions, len(word_to_ix),num_classes, 200, 2, True) # num layers was 1
    trainer = Trainer(
            batch_size,
            learning_rate,
            early_stop_count,
            epochs,
            model,
            dataloaders
        )
    trainer.train()
    trainer.report_final_loss()
    create_plots(trainer, "3_class_english")
