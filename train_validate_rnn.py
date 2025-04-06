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
from utils import load_config
from data import download_combined
import argparse



# Training loop for the model.
def train_model(model, train_loader, num_epochs=10, learning_rate=0.001, device="cpu"):
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


def train_autoregressive_rnn(
    sequence_length,
    input_size,
    output_size,
    train_data,
    hidden_size=64,
    num_layers=1,
    num_epochs=10,
    learning_rate=0.001,
    device="cpu",
):
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
    model = rnn_autoregressor.AutoregressiveRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    )

    # Train the model using the train_model function.
    train_model(
        model,
        train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    return model


def train_and_test_autoregressive_rnn(
    sequence_length,
    input_size,
    output_size,
    train_data,
    test_data,
    hidden_size=64,
    num_layers=1,
    num_epochs=10,
    learning_rate=0.001,
    device="cpu",
):
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
    model = rnn_autoregressor.AutoregressiveRNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size,
    )

    # Train the model using the train_model function.
    train_losses = train_model(
        model,
        train_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
    )

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


            # if you want to only test close price, use these instead
            # outputs_close = outputs[..., 0]
            # targets_close = targets[..., 0]
            # outputs_close = torch.tensor(outputs_close, dtype=torch.float32, device=device)
            # targets_close = torch.tensor(targets_close, dtype=torch.float32, device=device)
            # loss = criterion(outputs_close, targets_close)
            outputs_close = outputs[..., 0]
            targets_close = targets[..., 0]
            if args.print_predictions:
                print(
                    f"predicted vs actual close prices:{[f'{predicted} vs {actual}' for predicted, actual in zip(outputs_close, targets_close)]}"
                )
            outputs = torch.tensor(outputs, dtype=torch.float32, device=device)
            targets = torch.tensor(targets, dtype=torch.float32, device=device)
            loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)

        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)

        print(f"Test Loss: {test_loss}")

    return train_losses, test_losses


def rnn_prepare_train_test(
    sequence_length, hidden_size, num_layers, num_epochs, learning_rate, device
):
    default_route = experiment_folder
    data_path = os.path.join(default_route, f"combined_data_{config.dataset.start_date_train}_to_{config.dataset.corn_end_date_train}.csv")
    data = pd.read_csv(data_path)

    # Convert all columns to numeric, coercing errors to NaN
    data = data.apply(pd.to_numeric, errors="coerce")

    # Fill missing values by forward filling, backward filling, and rolling mean
    data = (
        data.fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(data.rolling(5, min_periods=1).mean())
    )

    alldata = data.values[:, 1:]
    print(f"alldata shape {alldata.shape}")
    num_features = alldata.shape[1]

    # normalize data using a standard scaler
    input_size = alldata.shape[1]
    output_size = input_size

    # Train and test the model.

    # Reshape data to fit TS2Vec expected input shape
    # Commonly, models expect [batch_size, sequence_length, feature_dim]
    # Here we treat the entire series as one batch with one feature.
    ts_input = alldata.reshape(1, -1, num_features)

    # Define TimeSeriesSplit parameters
    n_splits = config.training.n_splits  # Number of splits
    max_train_size = None  # Maximum size of the training dataset
    test_size = int(alldata.shape[0] * 0.1)
    gap = 0  # Number of samples to exclude between train and test sets
    tscv = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=max_train_size, test_size=test_size, gap=gap
    )

    # Initialize TimeSeriesSplit
    for split_idx, (train_index, test_index) in enumerate(tscv.split(ts_input[0])):
        train_data, test_data = ts_input[0][train_index], ts_input[0][test_index]

        # Example usage:
        train_sequences, train_targets = prepare_sequences_targets(
            train_data, sequence_length
        )
        test_sequences, test_targets = prepare_sequences_targets(
            test_data, sequence_length
        )

        # transforming the given sequences
        

        train_losses, test_losses = train_and_test_autoregressive_rnn(
            sequence_length,
            input_size,
            output_size,
            (train_sequences, train_targets),
            (test_sequences, test_targets),
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
        )


def rnn_prepare_train(
    sequence_length=30,
    hidden_size=64,
    num_layers=1,
    num_epochs=10,
    learning_rate=0.001,
    device="cpu",
):
    # Example parameters.
    default_route = experiment_folder
    data_path = os.path.join(
        default_route,
        f"combined_data_{config.dataset.start_date_train}_to_{config.dataset.corn_end_date_train}.csv",
    )
    data = pd.read_csv(data_path)

    # Convert all columns to numeric, coercing errors to NaN
    data = data.apply(pd.to_numeric, errors="coerce")

    # Fill missing values by forward filling, backward filling, and rolling mean
    data = (
        data.fillna(method="ffill")
        .fillna(method="bfill")
        .fillna(data.rolling(5, min_periods=1).mean())
    )

    alldata = data.values[:, 1:]
    print(f"alldata shape {alldata.shape}")
    num_features = alldata.shape[1]

    train_sequences, train_targets = prepare_sequences_targets(alldata, sequence_length)

    input_size = alldata.shape[1]
    output_size = input_size

    model = train_autoregressive_rnn(
        sequence_length=sequence_length,
        input_size=input_size,
        output_size=output_size,
        train_data=(train_sequences, train_targets),
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        device=device,
    )

    model.save_model(experiment_folder, model_name=config.experiment_name)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an autoregressive RNN model.")
    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="Print predicted vs actual values during testing.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file."
    )
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)
    config = load_config(args.config_path)
    experiment_folder = os.path.join("results", config.experiment_name + "_rnn")
    os.makedirs(experiment_folder, exist_ok=True)

    download_combined(config.dataset.start_date_train, config.dataset.corn_end_date_train, config.dataset.weather_end_date_train, ticker=config.dataset.ticker, keep_columns=config.dataset.features, save_path=experiment_folder)
    download_combined(config.dataset.start_date_test, config.dataset.corn_end_date_test, config.dataset.weather_end_date_test, ticker=config.dataset.ticker, keep_columns=config.dataset.features, save_path=experiment_folder)

    if config.mode == "train":
        rnn_prepare_train(
            sequence_length=config.training.sequence_length,
            hidden_size=config.training.hidden_size,
            num_layers=config.training.num_layers,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            device=args.device,
        )
    elif config.mode == "validate":
        rnn_prepare_train_test(
            sequence_length=config.training.sequence_length,
            hidden_size=config.training.hidden_size,
            num_layers=config.training.num_layers,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            device=args.device,
        )
