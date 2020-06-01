import torch
from torch import nn
from torch.nn import functional as F

class Glove_model(nn.Module):
    def __init__(self, dim, classes, lstm_units=200, num_layers=2, bidirectional=True):
        super(Glove_model, self).__init__()

        self.num_classes = classes

        self.lstm =  nn.LSTM(dim, lstm_units, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_units*(1+int(bidirectional)), classes)
        )

    def forward(self, x):
        # only use sequence for prediction, discard user id
        x, (h,c)= self.lstm(x[:,:-1])
        x = x.transpose(0,1)
        x = self.fc(x[-1])
        return x

class Embedding_Model(nn.Module):
    def __init__(self, dim, vocab_size, classes, lstm_units=100, num_layers=3, bidirectional=True):
        super(Embedding_Model, self).__init__()

        self.num_classes = classes

        self.word_embeddings = nn.Embedding(vocab_size, dim)
        self.lstm =  nn.LSTM(dim, lstm_units, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_units*(1+int(bidirectional)), classes)
        )

    def forward(self, x):
        # only use sequence for prediction, discard user id
        x = self.word_embeddings(x[:,:-1])
        x, (h,c)= self.lstm(x)
        # # # average pooling
        # avg_pool = torch.mean(x, 1)
        # # max pooling
        # max_pool, _ = torch.max(x,1)
        # x = torch.cat((max_pool, avg_pool), 1)
        # x = self.fc(x)
        x = x.transpose(0,1)
        x = self.fc(x[-1])
        return x