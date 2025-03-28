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
    def __init__(self, input_size=1, hidden_size=64, num_layers=1, output_size=1):
        """
        Args:
            input_size: number of features per time step (e.g., 1 for a univariate series)
            hidden_size: number of features in the hidden state
            num_layers: number of stacked LSTM layers
            output_size: dimension of the output (typically 1 for regression)
        """
        self.hyperparams = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'output_size': output_size
        }
        
        super(AutoregressiveRNN, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        # x: shape (batch_size, sequence_length, input_size)
        lstm_out, hidden = self.lstm(x, hidden)
        # Use the last time-step's output for prediction.
        last_out = lstm_out[:, -1, :]
        pred = self.fc(last_out)
        return pred, hidden
    
    def save_model(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_hyperparams': self.hyperparams,  # like input size, layer sizes, etc
        }, os.path.join(path, 'rnn_autoregressor.pth'))


#TODO: move this to a larger train.py file
# # Example usage:
# if __name__ == '__main__':
#         # Example parameters.
#         sequence_length = 30  # days
#         input_size = 7        # univariate series
#         output_size = 7
#         data = pd.read_csv('./Data/corn_futures_2000-07-17_to_2025-03-24.csv')

#         # Convert all columns to numeric, coercing errors to NaN
#         data = data.apply(pd.to_numeric, errors='coerce')

#         # Fill missing values by forward filling, backward filling, and rolling mean
#         data = data.fillna(method="ffill").fillna(method="bfill").fillna(data.rolling(5, min_periods=1).mean())

#         alldata = data.values[:, 1:]
#         print(f'alldata shape {alldata.shape}')
#         num_features = alldata.shape[1]

#         # Train and test the model.

#         # Reshape data to fit TS2Vec expected input shape
#         # Commonly, models expect [batch_size, sequence_length, feature_dim]
#         # Here we treat the entire series as one batch with one feature.
#         ts_input = alldata.reshape(1, -1, num_features)

#         # Define TimeSeriesSplit parameters
#         n_splits = 9  # Number of splits
#         max_train_size = None  # Maximum size of the training dataset
#         test_size = int(alldata.shape[0] * 0.1)
#         gap = 0  # Number of samples to exclude between train and test sets
#         tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)


#         # Initialize TimeSeriesSplit
#         for split_idx, (train_index, test_index) in enumerate(tscv.split(ts_input[0])):
#             train_data, test_data = ts_input[0][train_index], ts_input[0][test_index]
            
#             # Example usage:
#             train_sequences, train_targets = prepare_sequences_targets(train_data, sequence_length)
#             test_sequences, test_targets = prepare_sequences_targets(test_data, sequence_length)
            
#             train_losses, test_losses = train_and_test_model(
#                 sequence_length, input_size, output_size,
#                 (train_sequences, train_targets),
#                 (test_sequences, test_targets),
#                 hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'
#             )
            

