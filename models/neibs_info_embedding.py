from http.client import NON_AUTHORITATIVE_INFORMATION
from tkinter import N
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class Path(MessagePassing):
    '''
    Simple implementation of PathNet
    '''

    def __init__(self, feature_length, args, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        self.device = args.cuda
        self.dropout = args.dropout_pathnet
        super(Path, self).__init__()
        self.feature_length, self.hidden_size, self.wl \
            = feature_length, args.n_hidden, args.walk_len

        self.fc0 = torch.nn.Linear(feature_length *2, self.hidden_size)

        # self.params = self.get_lstm_params(self.feature_length * 2, self.feature_length * 2, self.feature_length, self.device)
        self.params = self.get_lstm_params(self.hidden_size, self.hidden_size, self.feature_length, self.device)
        # self.params = self.get_lstm_params(self.feature_length * 2, self.hidden_size, self.feature_length, self.device)

    def get_lstm_params(self, num_inputs, num_hiddens, num_outputs, device):
        def normal(shape):
            return torch.randn(size=shape, device=device) * 0.01

        def three():
            return (normal((num_inputs, num_hiddens)),
                    torch.zeros(num_hiddens, device=device))

        W_xi, b_i = three()
        W_xf, b_f = three()
        W_xo, b_o = three()
        W_xc, b_c = three()
        # output parameters
        W_hq = normal((num_hiddens, num_outputs))
        b_q = torch.zeros(num_outputs, device=device)

        params = [W_xi, b_i, W_xf, b_f, W_xo, b_o, W_xc, b_c, W_hq, b_q]
        for param in params:
            param.requires_grad_(True)
        return params

    def init_lstm_state(self, batch_size, num_hiddens, device):
        return (
            torch.zeros((batch_size, num_hiddens), device=device),
            torch.zeros((batch_size, num_hiddens), device=device))

    def lstm_time_entropy(self, params, In, TR, H=None, C=None):
        [W_xi, b_i, W_xf, b_f, W_xo, b_o, W_xc, b_c, W_hq, b_q] = params

        I = torch.sigmoid((In @ W_xi) + (In * TR))
        F = torch.sigmoid((In @ W_xf) + (In * TR))
        O = torch.sigmoid((In @ W_xo) + (In * TR))
        C_tilda = torch.tanh((In @ W_xc) + (In * TR))
        C = F * C_tilda + I * C_tilda
        H = O * torch.tanh(C)
        Y = (H @ W_hq) + b_q

        return Y, (H, C)

    def forward(self, X, T, neis, num_w, walk_len, neis_time, path_weight):
        path_weight = torch.unsqueeze(path_weight, -1)
        nei = X[neis]  # (1000,120,800)
        timestamps = T[neis_time]  # (1000,120,800)

        input = torch.cat((nei, timestamps), dim=2)

        # a, b = torch.chunk(nei, 2, dim=1)
        # c, d = torch.chunk(timestamps, 2, dim=1)
        # nei2 = torch.cat([(a * c - b * d), (a * d + b * c)], dim=1)

        input = self.fc0(input)

        Y, (H, C) = self.lstm_time_entropy(self.params, input, path_weight)
        Y = torch.mean(Y, dim=1)
        return Y  # shape(the number of all nodes , hidden_size) ==> shape(the number of current epoch, hidden_size)