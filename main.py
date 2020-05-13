import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from dataloader import load_data
from model import Test_model
from trainer import Trainer, create_plots

if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    learning_rate = 1e-3
    early_stop_count = 4
    dimensions = 100
    dataloaders = load_data(batch_size, dimensions = dimensions)
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
