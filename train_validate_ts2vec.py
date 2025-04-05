import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from data import prepare_sequences_targets
from ts2vec_autoregressor import TS2VecRegressor
import os
# from data import download_corn_futures_full_data
import argparse

scaler = StandardScaler()


def train_model(model, train_sequences, train_targets, num_epochs, n_iters):
    
    print(f"training for {num_epochs} epochs with {n_iters} train steps")
    losses = model.ts2vec.fit(train_sequences, n_epochs=num_epochs, n_iters=n_iters, verbose=True)

    sequence_repr = model.ts2vec.encode(
        train_sequences, encoding_window="full_series", batch_size=32
    )
    
    print(f"shape of representations:{sequence_repr.shape}")
    print(f"shape of targets:{train_targets.shape}")
    
    model.ridge.fit(sequence_repr, train_targets)

    return losses

# Autoregressive prediction: use the model's output as the next input.

def train_linear_model(
    input_size,
    encoding_dims,
    train_data,
    hidden_size=64,
    depth=3,
    temporal_unit=0,
    num_epochs=10,
    n_iters=200,
    learning_rate=0.001,
    device="cpu",
):
    """
    Only train TS2Vec linear model.

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
    train_sequences, train_targets = train_data  # TODO: Verify the shapes and values!!!

    # Initialize the model.
    model = TS2VecRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_norm=True,
        depth=depth,
        encoding_dims=encoding_dims,
        encoder_weights=None,
        temporal_unit=temporal_unit,
        encoder_lr=learning_rate,
        device=device,
    )

    losses = train_model(model, train_sequences, train_targets, num_epochs, n_iters)

    print("Logging losses...")
    for idx, loss in enumerate(losses):
        print(f"Epoch {idx}: {loss}")
    
    return model


def train_and_test_linear_model(
    input_size,
    encoding_dims,
    train_data,
    test_data,
    hidden_size=64,
    depth=3,
    temporal_unit=0,
    num_epochs=10,
    n_iters=200,
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
    print("is test sequences a tensor?")
    print(isinstance(test_sequences, torch.Tensor))
    # Create a DataLoader for the test data
    test_dataset = [(seq, target) for seq, target in zip(test_sequences, test_targets)]
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the model.
    model = TS2VecRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        batch_norm=True,
        depth=depth,
        encoding_dims=encoding_dims,
        encoder_weights=None,
        temporal_unit=temporal_unit,
        encoder_lr=learning_rate,
        device=device,
    )
    print(f"train sequence dataset shape: {train_sequences.shape}")
    losses = train_model(model, train_sequences, train_targets, num_epochs, n_iters)

    # Testing phase.
    criterion = nn.MSELoss()
    test_losses = []

    with torch.no_grad():
        test_loss = 0.0
        for inputs, targets in test_loader:

            inputs = inputs.to(device)
            targets = targets.to(device)

            # do an inverse transform to allow comparison with raw data
            # outputs = scaler.inverse_transform(outputs.cpu().numpy().reshape(-1, outputs.shape[-1])).reshape(outputs.shape)
            # targets = scaler.inverse_transform(targets.cpu().numpy().reshape(-1, targets.shape[-1])).reshape(targets.shape)

            # if you want to only test close price, use these instead
            outputs = model.predict(inputs.to('cpu').numpy())

            outputs_close = outputs[..., 0]
            targets_close = targets[..., 0]
            if args.print_predictions:
                print(f"predicted vs actual close prices:{[f'{predicted} vs {actual}' for predicted, actual in zip(outputs_close, targets_close)]}")
            outputs_close = torch.tensor(outputs_close, dtype=torch.float32, device=device)
            targets_close = torch.tensor(targets_close, dtype=torch.float32, device=device)
            
            
            
            loss = criterion(outputs_close, targets_close)

            # Otherwise, use these
            # outputs = torch.tensor(outputs, dtype=torch.float32, device=device)
            # targets = torch.tensor(targets, dtype=torch.float32, device=device)
            # loss = criterion(outputs, targets)

            test_loss += loss.item()
            print(f"Test loss: {loss.item()}")

        test_loss /= len(test_loader.dataset)

        print(f"Test Loss: {test_loss}")

    return test_losses


def ts2vec_prepare_train_test(
    sequence_length=30,
    hidden_size=64,
    encoding_dims=128,
    num_epochs=10,
    n_iters=200,
    learning_rate=0.001,
    device="cpu",
):
    # Example parameters.
    default_route = "./Data/"
    data = pd.read_csv(default_route + args.dataset)

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

    input_size = alldata.shape[1]
    output_size = input_size

    # Reshape data to fit TS2Vec expected input shape
    ts_input = alldata.reshape(1, -1, num_features)

    # Define TimeSeriesSplit parameters
    n_splits = args.n_splits  # Number of splits
    test_size = int(alldata.shape[0] * 0.1)
    gap = 0  # Number of samples to exclude between train and test sets
    tscv = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=None, test_size=test_size, gap=gap
    )

    # Initialize TimeSeriesSplit
    for split_idx, (train_index, test_index) in enumerate(tscv.split(ts_input[0])):
        train_data, test_data = ts_input[0][train_index], ts_input[0][test_index]

        # Prepare sequences and targets for train and test data
        train_data = prepare_sequences_targets(train_data, sequence_length)
        test_data = prepare_sequences_targets(test_data, sequence_length)

        # Train and test the model
        train_and_test_linear_model(
            input_size=input_size,
            encoding_dims=encoding_dims,
            train_data=train_data,
            test_data=test_data,
            hidden_size=hidden_size,
            depth=3,
            temporal_unit=0,
            num_epochs=num_epochs,
            n_iters=n_iters,
            learning_rate=learning_rate,
            device=device,
        )


def ts2vec_prepare_train(
    sequence_length=30,
    hidden_size=64,
    encoding_dims=128,
    num_epochs=10,
    n_iters=200,
    learning_rate=0.001,
    device="cpu",
):
    # Example parameters.
    default_route = "./Data/"
    data = pd.read_csv(default_route + args.dataset)

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

    model = train_linear_model(
        input_size=input_size,
        encoding_dims=encoding_dims,
        train_data=(train_sequences, train_targets),
        hidden_size=hidden_size,
        num_epochs=num_epochs,
        n_iters=n_iters,
        learning_rate=learning_rate,
        device=device,
    )

    save_path = "ts2vec_models"
    os.makedirs(save_path, exist_ok=True)

    model.save_model(save_path)
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train an autoregressive RNN model.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "validate"],
        default="train",
        help="Mode to run the script: 'train' or 'validate'."
    )
    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="Print predicted vs actual values during testing."
    )
    parser.add_argument(
        "--sequence_length", type=int, default=30, help="Length of input sequences."
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=128,
        help="Number of features in the hidden state.",
    )
    parser.add_argument(
        "--encoding_dims",
        type=int,
        default=128,
        help="Dimension of the encoding representations.",
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--n_iters", type=int, default=200, help="Number of iterations per epoch."
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=3,
        help="Number of splits for TimeSeriesSplit during validation."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="'cpu' or 'cuda'.")
    parser.add_argument("--dataset", type=str, help="Path to the dataset CSV file.")

    args = parser.parse_args()
 

    # rnn_prepare_train(
    #     sequence_length=args.sequence_length,
    #     hidden_size=args.hidden_size,
    #     num_layers=args.num_layers,
    #     num_epochs=args.num_epochs,
    #     learning_rate=args.learning_rate,
    #     device=args.device
    # )
    
    
    if args.mode == "train":
        ts2vec_prepare_train(
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            encoding_dims=args.encoding_dims,
            num_epochs=args.num_epochs,
            n_iters=args.n_iters,
            learning_rate=args.learning_rate,
            device=args.device
        )
    elif args.mode == "validate":
        ts2vec_prepare_train_test(
            sequence_length=args.sequence_length,
            hidden_size=args.hidden_size,
            encoding_dims=args.encoding_dims,
            num_epochs=args.num_epochs,
            n_iters=args.n_iters,
            learning_rate=args.learning_rate,
            device=args.device
        )
