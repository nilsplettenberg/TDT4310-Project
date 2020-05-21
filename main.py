import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import pickle

from dataloader import load_data, get_data_loader
from model import Test_model, Embedding_Model
from trainer import Trainer, create_plots
from preprocess import preprocess, prepare_sequence, zero_pad

if __name__ == "__main__":
    epochs = 100
    batch_size = 128
    learning_rate = 1e-3
    early_stop_count = 5
    dimensions = 200
    num_classes = 3

    # x_train, y_train = load_data("data/pan19-author-profiling-20200229/training/en/", num_classes)
    # x_test, y_test = load_data("data/pan19-author-profiling-20200229/test/en/", num_classes)
    
    # # x_train, y_train = preprocess(x_train, y_train, dimensions)
    # # x_test, y_test = preprocess(x_test, y_test, dimensions)

    # # # without glove
    # # concatenate sets for preprocessing
    # x = x_train + x_test
    # y = y_train + y_test
    # x,y = (x, y)
    # x, word_to_ix = prepare_sequence(x)
    # x, y = zero_pad(x,y)

    # # x_test, word_to_ix = prepare_sequence(x_test, word_to_ix)
    # # x_test, y_test = zero_pad(x_test,y_test)
    # # datasets = (x_train, y_train, x_test, y_test)
    # datasets = (x,y)
    # # Saving the objects:
    # with open('/work/nilsple/data/preprocessed.pkl', 'wb') as f: 
    #     pickle.dump((datasets, word_to_ix), f)

    # Getting preprocessed datasets
    with open('/work/nilsple/data/preprocessed.pkl', 'rb') as f:
        datasets, word_to_ix = pickle.load(f)

    dataloaders = get_data_loader(datasets, batch_size, dimensions = dimensions)
    del datasets
    # model = Test_model(dimensions,num_classes, 100, 3)
    model = Embedding_Model(dimensions, len(word_to_ix),num_classes, 100, 2)
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
    create_plots(trainer, "test run")
