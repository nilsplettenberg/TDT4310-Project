import torch
from torch import nn
from torch.nn import functional as F

class Test_model(nn.Module):
    def __init__(self, input_dim, classes, lstm_units=200):
        super(Test_model, self).__init__()

        # lstm don't support nn.Sequential because the return hidden states
        # self.lstm_units = lstm_units
        # self.lstm1 = nn.LSTM(input_dim, lstm_units[0], bidirectional=True, batch_first=True)
        # self.lstm2 = nn.LSTM(lstm_units[0], lstm_units[1], bidirectional=True, batch_first=True)
        # self.lstm3 = nn.LSTM(lstm_units[1], lstm_units[2], bidirectional=True, batch_first=True)
        self.lstm =  nn.LSTM(input_dim, lstm_units, num_layers=3, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(lstm_units*2, classes)

    def forward(self, x):
        # x, _= self.lstm1(x)
        # x, _= self.lstm2(x)
        # x, _= self.lstm3(x)
        x, _= self.lstm(x)
        x = self.fc(x)
        return x
