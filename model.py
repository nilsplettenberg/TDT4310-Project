import torch
from torch import nn
from torch.nn import functional as F

class Test_model(nn.Module):
    def __init__(self, input_dim, classes, lstm_units=100):
        super(Test_model, self).__init__()

        self.lstm =  nn.LSTM(input_dim, lstm_units, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(lstm_units*2, classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x, _= self.lstm(x)
        # # average pooling
        # avg_pool = torch.mean(x, 1)
        # # max pooling
        # max_pool, _ = torch.max(x,1)
        # x = torch.cat((max_pool, avg_pool), 1)
        # x = self.fc(x)
        x = x.transpose(0,1)
        x = self.fc(x[-1])
        return x
