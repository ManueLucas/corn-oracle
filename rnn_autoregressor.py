import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

# Training loop for the model.
def train_model(model, train_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * inputs.size(0)
            
        epoch_loss /= len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Autoregressive prediction: use the model's output as the next input.
def autoregressive_prediction(model, initial_sequence, prediction_steps=1, device='cpu'):
    """
    Generates predictions in an autoregressive fashion.
    
    Args:
        model: trained autoregressive model.
        initial_sequence: torch.Tensor of shape (sequence_length, input_size)
        prediction_steps: number of future time steps to forecast.
        device: 'cpu' or 'cuda'
    
    Returns:
        predictions: torch.Tensor of shape (prediction_steps, output_size)
    """
    model.eval()
    predictions = []
    # Add batch dimension: shape becomes (1, sequence_length, input_size)
    current_seq = initial_sequence.unsqueeze(0).to(device)
    
    with torch.no_grad():
        for _ in range(prediction_steps):
            pred, _ = model(current_seq)
            predictions.append(pred.squeeze(0))  # remove batch dimension
            # Append prediction to sequence and drop the oldest time step.
            new_input = pred.unsqueeze(1)  # shape: (1, 1, input_size)
            current_seq = torch.cat((current_seq[:, 1:, :], new_input), dim=1)
    
    return torch.stack(predictions)

    # Train and test functions to track accuracy and loss.
def train_and_test_model(sequence_length, input_size, output_size, train_data, test_data, hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Train and test the autoregressive RNN model, tracking accuracy and loss.

    Args:
        sequence_length: length of input sequences.
        input_size: number of features per time step.
        output_size: dimension of the output.
        train_data: tuple (train_sequences, train_targets) as numpy arrays.
        test_data: tuple (test_sequences, test_targets) as numpy arrays.
        hidden_size: number of features in the hidden state.
        num_layers: number of stacked LSTM layers.
        num_epochs: number of training epochs.
        learning_rate: learning rate for the optimizer.
        device: 'cpu' or 'cuda'.

    Returns:
        train_losses, test_losses: lists of loss values for each epoch.
    """
    # Convert train and test data to torch tensors.
    train_sequences, train_targets = train_data
    test_sequences, test_targets = test_data

    train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    test_sequences = torch.tensor(test_sequences, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    # Create datasets and data loaders.
    train_dataset = TensorDataset(train_sequences, train_targets)
    test_dataset = TensorDataset(test_sequences, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model.
    model = AutoregressiveRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)
    model.to(device)

    # Define loss function and optimizer.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []

    for epoch in range(num_epochs):
        # Training phase.
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Testing phase.
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses

def prepare_sequences_targets(data, sequence_length):
            """
            Prepares sequences and corresponding targets for training and testing.

            Args:
                data: numpy array of shape (num_samples, num_features).
                sequence_length: length of input sequences.

            Returns:
                sequences: numpy array of shape (num_samples, sequence_length, input_size).
                targets: numpy array of shape (num_samples, output_size).
            """
            print(data.shape)
            num_samples = data.shape[0]
            sequences = []
            targets = []

            for i in range(num_samples - sequence_length):
                seq = data[i:i + sequence_length]
                target = data[i + sequence_length]  # the next time step
                sequences.append(seq)
                targets.append(target)

            sequences = np.array(sequences)  # shape: (num_samples, sequence_length, input_size)
            targets = np.array(targets)       # shape: (num_samples, output_size)

            return sequences, targets
# Example usage:
if __name__ == '__main__':
        # Example parameters.
        sequence_length = 30  # days
        input_size = 7        # univariate series
        output_size = 7
        data = pd.read_csv('./Data/corn_futures_2000-07-17_to_2025-03-24.csv')

        # Convert all columns to numeric, coercing errors to NaN
        data = data.apply(pd.to_numeric, errors='coerce')

        # Fill missing values by forward filling, backward filling, and rolling mean
        data = data.fillna(method="ffill").fillna(method="bfill").fillna(data.rolling(5, min_periods=1).mean())

        alldata = data.values[:, 1:]
        print(f'alldata shape {alldata.shape}')
        num_features = alldata.shape[1]

        # Train and test the model.

        # Reshape data to fit TS2Vec expected input shape
        # Commonly, models expect [batch_size, sequence_length, feature_dim]
        # Here we treat the entire series as one batch with one feature.
        ts_input = alldata.reshape(1, -1, num_features)

        # Define TimeSeriesSplit parameters
        n_splits = 9  # Number of splits
        max_train_size = None  # Maximum size of the training dataset
        test_size = int(alldata.shape[0] * 0.1)
        gap = 0  # Number of samples to exclude between train and test sets
        tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap)


        # Initialize TimeSeriesSplit
        for split_idx, (train_index, test_index) in enumerate(tscv.split(ts_input[0])):
            train_data, test_data = ts_input[0][train_index], ts_input[0][test_index]
            
            # Example usage:
            train_sequences, train_targets = prepare_sequences_targets(train_data, sequence_length)
            test_sequences, test_targets = prepare_sequences_targets(test_data, sequence_length)
            
            train_losses, test_losses = train_and_test_model(
                sequence_length, input_size, output_size,
                (train_sequences, train_targets),
                (test_sequences, test_targets),
                hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'
            )
            

