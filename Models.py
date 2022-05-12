import torch
import torch.nn.functional as F
from torch import nn


class RNNClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, out, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNNCell(input_dim, hidden_dim)  # RNN Cell
        # fully connected layer: maps last hidden vector to model prediction
        self.fc = nn.Linear(hidden_dim, out)
        self.activation = nn.Sigmoid()                # coz binary classification
        self.device = device

    def forward(self, x):

        hidden = self.init_hidden(x)

        #############################
        # shape of x is (batches,time_Steps,features)
        time_steps = x.shape[1]

        for i in range(0, time_steps):
            inputs = x[:, i]                     # (batch,features) shape
            hidden = self.rnn(inputs, hidden)

        # take the hidden vector corresponding to last time step
        out = self.fc(hidden)
        ###########################

        return self.activation(out)

    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim)
        return h0.to(self.device)


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, out, device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTMCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, out)
        self.activation = nn.Sigmoid()
        self.device = device

    def forward(self, x):
        hidden, cell = self.init_hidden(x)

        ############################
        time_steps = x.shape[1]  # shape of x is (batches,time_Steps,features)

        for i in range(time_steps):
            inputs = x[:, i]                  # (batch,features) shape
            hidden, cell = self.rnn(inputs, (hidden, cell))

        # take the hidden vector corresponding to last time step
        out = self.fc(hidden)
        #############################

        return self.activation(out)

    def init_hidden(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim, device=self.device)
        c0 = torch.zeros(x.size(0), self.hidden_dim, device=self.device)
        return h0, c0
