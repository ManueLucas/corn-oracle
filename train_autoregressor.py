import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from data import prepare_sequences_targets
import rnn_autoregressor
import os
from data import download_corn_futures_full_data
import argparse

# Training loop for the model.
def train_model(model, train_loader, num_epochs=10, learning_rate=0.001, device='cpu'):
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    epoch_losses = []
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
        epoch_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    return epoch_losses

# Autoregressive prediction: use the model's output as the next input.

def train_autoregressive_rnn(sequence_length, input_size, output_size, train_data, hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'):
    """
    Only train autoregressive RNN model, tracking accuracy and loss.

    Args:
        sequence_length: length of input sequences.
        input_size: number of features per time step.
        output_size: dimension of the output.
        train_data: tuple (train_sequences, train_targets) as numpy arrays.
        hidden_size: number of features in the hidden state.
        num_layers: number of stacked LSTM layers.
        num_epochs: number of training epochs.
        learning_rate: learning rate for the optimizer.
        device: 'cpu' or 'cuda'.

    Returns:
        model: trained model.
    """
    # Convert train data to torch tensors.
    train_sequences, train_targets = train_data
    train_sequences = torch.tensor(train_sequences, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)

    # Create dataset and data loader.
    train_dataset = TensorDataset(train_sequences, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize the model.
    model = rnn_autoregressor.AutoregressiveRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

    # Train the model using the train_model function.
    train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    return model


def train_and_test_autoregressive_rnn(sequence_length, input_size, output_size, train_data, test_data, hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'):
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
    model = rnn_autoregressor.AutoregressiveRNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size)

    # Train the model using the train_model function.
    train_losses = train_model(model, train_loader, num_epochs=num_epochs, learning_rate=learning_rate, device=device)

    # Testing phase.
    model.eval()
    criterion = nn.MSELoss()
    test_losses = []
    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs, _ = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f"Test Loss: {test_loss}")

    return train_losses, test_losses


def rnn_prepare_train_test():
    # Example parameters.
    sequence_length = 30  # days
    # input_size = 7       
    # output_size = 7
    default_route = "./Data/"
    data = pd.read_csv(default_route + args.dataset)

    # Convert all columns to numeric, coercing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill missing values by forward filling, backward filling, and rolling mean
    data = data.fillna(method="ffill").fillna(method="bfill").fillna(data.rolling(5, min_periods=1).mean())

    alldata = data.values[:, 1:]
    print(f'alldata shape {alldata.shape}')
    num_features = alldata.shape[1]

    # normalize data using a standard scaler
    alldata = StandardScaler().fit_transform(alldata)
    
    input_size =  alldata.shape[1]
    output_size = input_size

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
        
        train_losses, test_losses = train_and_test_autoregressive_rnn(
            sequence_length, input_size, output_size,
            (train_sequences, train_targets),
            (test_sequences, test_targets),
            hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'
        )
        
def rnn_prepare_train(sequence_length=30, hidden_size=64, num_layers=1, num_epochs=10, learning_rate=0.001, device='cpu'): 
    # Example parameters.
    default_route = "./Data/"
    data = pd.read_csv(default_route + args.dataset)

    # Convert all columns to numeric, coercing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    # Fill missing values by forward filling, backward filling, and rolling mean
    data = data.fillna(method="ffill").fillna(method="bfill").fillna(data.rolling(5, min_periods=1).mean())

    alldata = data.values[:, 1:]
    print(f'alldata shape {alldata.shape}')
    num_features = alldata.shape[1]
    
    train_sequences, train_targets = prepare_sequences_targets(alldata, sequence_length)

    input_size =  alldata.shape[1]
    output_size = input_size

    model = train_autoregressive_rnn(sequence_length, input_size, output_size, (train_sequences, train_targets), hidden_size=hidden_size, num_layers=num_layers, num_epochs=num_epochs, learning_rate=learning_rate, device=device)
    
    save_path = "checkpoints"
    os.makedirs(save_path, exist_ok=True)
    
    model.save_model(save_path)
    return model
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an autoregressive RNN model.")
    parser.add_argument("--sequence_length", type=int, default=30, help="Length of input sequences.")
    parser.add_argument("--hidden_size", type=int, default=64, help="Number of features in the hidden state.")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of stacked LSTM layers.")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer.")
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset CSV file.")

    args = parser.parse_args()
    print("Hyperparameters:")
    print(f"Sequence Length: {args.sequence_length}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Number of Layers: {args.num_layers}")
    print(f"Number of Epochs: {args.num_epochs}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    
    # rnn_prepare_train(
    #     sequence_length=args.sequence_length,
    #     hidden_size=args.hidden_size,
    #     num_layers=args.num_layers,
    #     num_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     device=args.device
    # )

    rnn_prepare_train_test()