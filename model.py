import torch
import torch.nn as nn

class stock_model(nn.Model):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True)
        self.output_layer = nn.Linear(hidden_size)