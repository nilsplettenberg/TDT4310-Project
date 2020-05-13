import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F

from preprocess import embed_words, zero_pad
from dataloader import load_data
from model import Test_model
from trainer import Trainer

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

# x_train, y_train = loadData("data/pan19-author-profiling-20200229/training/en/")
# x_test, y_test = loadData("data/pan19-author-profiling-20200229/test/en/")

# # embedd words using glove matrix
# x_train = embed_words(x_train[:5])
# x_test = embed_words(x_test[:5])
# # remove too short and too long sequences, padd with zeros 
# x_train = zero_pad(x_train,y_train)
# x_test = zero_pad(x_test,y_test)

# # convert to torch tensor
# x_train = torch.tensor(x_train)
# y_train = F.one_hot(torch.tensor(y_train))
# x_test = torch.tensor(x_test)
# y_test = F.one_hot(torch.tensor(y_test))

# train_dataset = data.TensorDataset(x_train, y_train)
# test_dataset = data.TensorDataset(x_test)
if __name__ == "__main__":
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_data(batch_size)
    model = Test_model(25,3)
    trainer = Trainer(
            batch_size,
            learning_rate,
            early_stop_count,
            epochs,
            model,
            dataloaders
        )
    trainer.train()