import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit


# Define the autoregressive RNN model using LSTM.
class AutoregressiveRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1, batch_norm=True, intermediate_size=32):
        """
        Args:
            input_size: number of features per time step (e.g., 1 for a univariate series)
            hidden_size: number of features in the hidden state
            num_layers: number of stacked LSTM layers
            output_size: dimension of the output (typically 1 for regression)
            intermediate_size: size of the intermediate dense layer
        """
        self.hyperparams = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size,
            'intermediate_size': intermediate_size
        }

        super(AutoregressiveRNN, self).__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_normalization_layer = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # Add an intermediate dense layer with ReLU activation
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(intermediate_size, output_size)

    def forward(self, x, hidden=None):
        # Normalize input data
        if self.batch_norm:
            x = self.batch_normalization_layer(x)

        # x: shape (batch_size, sequence_length, input_size)
        lstm_out, hidden = self.lstm(x, hidden)

        # Use the last time-step's output for prediction.
        last_out = lstm_out[:, -1, :]  # shape: (batch_size, hidden_size)

        # Pass through dense layers with ReLU
        x = self.fc1(last_out)
        x = self.relu(x)
        pred = self.fc2(x)
        return pred, hidden

    def save_model(self, path, model_name=None):
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        # Save the model with the specified filename
        extension = '.pth'
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_hyperparams': self.hyperparams,
        }, os.path.join(path, f"{model_name}{extension}"))
