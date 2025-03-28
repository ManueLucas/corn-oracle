import argparse
from rnn_autoregressor import AutoregressiveRNN
import torch
from data import prepare_sequences_targets
import pandas as pd
import numpy as np
from data import download_corn_futures_eval_data

def load_model(model_name, checkpoint_name, device):
    """
    Load the model and checkpoint.
    Placeholder function to be implemented.
    """
    # TODO: Add logic to load the specified model and checkpoint
    checkpoints_folder = "checkpoints/"
    checkpoint = torch.load(f"{checkpoints_folder}{checkpoint_name}")

    params = checkpoint['model_hyperparams']


    if model_name == "AutoregressiveRNN":
        model = AutoregressiveRNN(**params).to(device)
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
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

def eval_raw(model, device):
    sequence_length = 30  # days
    input_size = 7        # univariate series
    output_size = 7
    data = pd.read_csv('./Data/eval_corn_futures_2025-01-01_to_2025-03-21.csv')
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
    print(f'total length of data: {alldata.shape[0]}')
    print(f'initial_sequence length: {initial_sequence.shape[0]}')
    
    initial_sequence = torch.tensor(initial_sequence, dtype=torch.float).to(device)
    actual = torch.tensor(actual, dtype=torch.float).to(device)
    
    prediction = autoregressive_prediction(model=model, initial_sequence=initial_sequence, prediction_steps=num_steps_to_predict, device=device)
    predicted_close = prediction[:, 0]
    
    import matplotlib.pyplot as plt

    # Offset predicted close by the sequence length
    offset = sequence_length
    predicted_close_offset = [np.nan] * offset + predicted_close.tolist()
    print(dates)
    print(predicted_close_offset)
    print(actual_close)
    print(dates.shape)
    print(len(predicted_close_offset))
    print(len(actual_close))
    
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
    
    


def evaluate_model(model, features, device):
    """
    Evaluate the model.
    Placeholder function to be implemented.
    """
    # TODO: Add logic to evaluate the model (add separate logic for each model and the type of features they use, whether they require preprocessing or some unsupervised step before)
    if features == "raw":
        download_corn_futures_eval_data()
        eval_raw(model, device)
        

def main():
    parser = argparse.ArgumentParser(description="Evaluate a model with a given checkpoint.")
    parser.add_argument("--model", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--features", type=str, required=True, help="Subset of features that the model was trained on.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the evaluation on ('cpu' or 'cuda').")
    args = parser.parse_args()

    # Load the model and checkpoint
    model = load_model(args.model, args.checkpoint, args.device)

    # Evaluate the model
    evaluate_model(model, args.features, args.device)

if __name__ == "__main__":
    main()