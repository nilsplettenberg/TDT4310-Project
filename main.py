import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
import pickle

from dataloader import load_data, get_data_loader
from model import Test_model
from trainer import Trainer, create_plots
from preprocess import preprocess

if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    learning_rate = 1e-3
    early_stop_count = 4
    dimensions = 100

    # x_train, y_train = load_data("data/pan19-author-profiling-20200229/training/en/")
    # x_test, y_test = load_data("data/pan19-author-profiling-20200229/test/en/")

    # x_train, y_train = preprocess(x_train, y_train, dimensions)
    # x_test, y_test = preprocess(x_test, y_test, dimensions)

    # datasets = (x_train, y_train, x_test, y_test)
    # # Saving the objects:
    # with open('/work/nilsple/data/preprocessed.pkl', 'wb') as f: 
    #     pickle.dump(datasets, f)

    # Getting preprocessed datasets
    with open('/work/nilsple/data/preprocessed.pkl', 'rb') as f:
        datasets = pickle.load(f)

    dataloaders = get_data_loader(datasets, batch_size, dimensions = dimensions)
    del datasets
    model = Test_model(dimensions,3)
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
