import argparse
from rnn_autoregressor import AutoregressiveRNN
from ts2vec_autoregressor import TS2VecRegressor
import torch
from data import prepare_sequences_targets
import pandas as pd
import numpy as np
import os
from utils import load_config
from data import download_corn_futures_eval_data, download_combined

def load_model(model_name, device):
    """
    Load the model and checkpoint.
    Placeholder function to be implemented.
    """
    # TODO: Add logic to load the specified model and checkpoint
    
    checkpoint_path = os.path.join("results", config.experiment_name + f"_{config.model_type}", config.experiment_name + ".pth")

    if model_name == "rnn":
        checkpoint = torch.load(f"{checkpoint_path}")

        params = checkpoint['model_hyperparams']
        model = AutoregressiveRNN(**params).to(device)
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    elif model_name == "ts2veclinear":
        checkpoint = torch.load(f"{checkpoint_path}")

        params = checkpoint['model_hyperparams']
        model = TS2VecRegressor(**params)

        
    return model

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

def autoregressive_prediction_ts2vec(model, initial_sequence, prediction_steps=1):
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
    predictions = []
    # (sequence_length, features)
    current_seq = np.expand_dims(initial_sequence, axis=0)
    with torch.no_grad():
        for _ in range(prediction_steps):
            pred = model.predict(current_seq) # pred should be (features)
            predictions.append(pred)
            # Append prediction to sequence
            new_input = np.expand_dims(pred, axis=0)  # shape: (1, features)
            current_seq = np.concatenate((current_seq[:, 1:, :], new_input), axis=1) # exclude last timestep, append new prediction
    return np.stack(predictions)

def eval_rnn(model, device, data):
    sequence_length = 30  # days
    input_size = 7        # univariate series
    output_size = 7
    #data = pd.read_csv('./Data/eval_corn_futures_2025-01-01_to_2025-03-31.csv')
    # Convert all columns to numeric, coercing errors to NaN
    # Convert all other columns to numeric (floats), coercing errors to NaN
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
    dates = data.iloc[:, 0]


    # Fill missing values by forward filling, backward filling, and rolling mean
    data.iloc[:, 1:] = data.iloc[:, 1:].fillna(method="ffill").fillna(method="bfill").fillna(data.iloc[:, 1:].rolling(5, min_periods=1).mean())

    # Extract dates for matplotlib
    
    alldata = data.values[:, 1:].astype(np.float32)
    actual_close = alldata[:, 0]
    
    num_features = alldata.shape[1]
    sequences, actual = prepare_sequences_targets(alldata, sequence_length)
    initial_sequence = sequences[0]
    num_steps_to_predict = actual.shape[0]
    
    initial_sequence = torch.tensor(initial_sequence, dtype=torch.float).to(device)
    actual = torch.tensor(actual, dtype=torch.float).to(device)
    
    prediction = autoregressive_prediction(model=model, initial_sequence=initial_sequence, prediction_steps=num_steps_to_predict, device=device)
    predicted_close = prediction[:, 0]
    
    import matplotlib.pyplot as plt

    # Offset predicted close by the sequence length
    offset = sequence_length
    predicted_close = predicted_close.cpu().numpy()
    actual_close_offset = actual_close[offset:]
    print(f"MAE for prediction vs actual: {np.mean(np.abs(predicted_close - actual_close_offset))}")
    predicted_close_offset = [np.nan] * offset + predicted_close.tolist()    
    # Plot actual and predicted close prices
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_close, label="Actual Close", color="blue")
    plt.plot(dates, predicted_close_offset, label="Predicted Close", color="orange", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Actual vs Predicted Close Prices")
    plt.legend()
    plt.grid()

    # Save the figure
    plt.savefig("predicted_vs_actual_close.png")
    plt.show()
    
    
    

def eval_ts2vec(model, device, data):
    sequence_length = 30  # days
    # Convert all columns to numeric, coercing errors to NaN
    # Convert all other columns to numeric (floats), coercing errors to NaN
    data.iloc[:, 1:] = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    
    data.iloc[:, 0] = pd.to_datetime(data.iloc[:, 0], format='%Y-%m-%d', errors='coerce')
    dates = data.iloc[:, 0]


    # Fill missing values by forward filling, backward filling, and rolling mean
    data.iloc[:, 1:] = data.iloc[:, 1:].fillna(method="ffill").fillna(method="bfill").fillna(data.iloc[:, 1:].rolling(5, min_periods=1).mean())
    
    alldata = data.values[:, 1:].astype(np.float32)
    actual_close = alldata[:, 0]
    
    num_features = alldata.shape[1]
    sequences, actual = prepare_sequences_targets(alldata, sequence_length)
    initial_sequence = sequences[0]
    num_steps_to_predict = actual.shape[0]
    print(f'total length of data: {alldata.shape[0]}')
    print(f'initial_sequence length: {initial_sequence.shape[0]}')

    prediction = autoregressive_prediction_ts2vec(model=model, initial_sequence=initial_sequence, prediction_steps=num_steps_to_predict)
    predicted_close = prediction[:, :, 0].squeeze(-1)
    
    import matplotlib.pyplot as plt


    # Offset predicted close by the sequence length
    offset = sequence_length
    predicted_close_offset = [np.nan] * offset + predicted_close.tolist()
    # print(dates)
    # print(predicted_close_offset)
    # print(actual_close)
    # print(dates.shape)
    # print(len(predicted_close_offset))
    # print(len(actual_close))
    
    # Plot actual and predicted close prices
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual_close, label="Actual Close", color="blue")
    plt.plot(dates, predicted_close_offset, label="Predicted Close", color="orange", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title("Actual vs Predicted Close Prices")
    plt.legend()
    plt.grid()

    # Save the figure
    plt.savefig("predicted_vs_actual_close_ts2vec.png")
    plt.show()

def evaluate_model(model, device):
    """
    Evaluate the model.
    """
    data_path = f'{experiment_folder}/combined_data_{config.dataset.start_date_test}_to_{config.dataset.corn_end_date_test}.csv'
    print(f"Loading data from {data_path}")
    data = pd.read_csv(f'{experiment_folder}/combined_data_{config.dataset.start_date_test}_to_{config.dataset.corn_end_date_test}.csv')

    if model.__class__.__name__ == "TS2VecRegressor":
        eval_ts2vec(model=model, device=device, data=data)
    elif model.__class__.__name__ == "AutoregressiveRNN":
        eval_rnn(model=model, device=device, data=data)
    else:
        raise ValueError(f"Unsupported model class: {model.__class__.__name__}")
        
        
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a given checkpoint.")
    parser.add_argument("--config", type=str, required=True, help="Configuration file for the model.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default='cpu', help="Device to run the model on: 'cuda' or 'cpu'.")
    
    args = parser.parse_args()
    os.makedirs("results", exist_ok=True)

    config = load_config(args.config)
    
    experiment_folder = os.path.join("results", config.experiment_name + f"_{config.model_type}")

    # Load the model and checkpoint
    model = load_model(config.model_type, args.device)

    # Evaluate the model
    evaluate_model(model, args.device)
