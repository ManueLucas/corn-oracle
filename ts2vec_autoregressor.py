import torch
import torch.nn as nn
import os
from ts2vec import TS2Vec
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
import pickle



class TS2VecRegressor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, batch_norm=True, depth=3, encoding_dims=1, encoder_weights=None, regressor_weights=None, temporal_unit=0, encoder_lr=0.001, device='cpu'):
        """
        Args:
            input_size: number of features per time step (e.g., 1 for a univariate series)
            hidden_size: number of ts2vec hidden layer params
            depth: number of ts2vec layers
            temporal_unit: temporal resolution of encoding
            num_layers: number of stacked LSTM layers
            output_size: dimension of the output (typically 1 for regression)
        """
        super(TS2VecRegressor, self).__init__()
        self.hyperparams = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'batch_norm': batch_norm,
            'depth': depth,
            'encoding_dims': encoding_dims,
            'encoder_weights': encoder_weights,
            'regressor_weights': regressor_weights,
            'temporal_unit': temporal_unit,
            'encoder_lr': encoder_lr,
            'device': device,
        }
        self.ts2vec = TS2Vec(input_dims=input_size, output_dims=encoding_dims, hidden_dims=hidden_size, depth=depth, device=device, lr=encoder_lr, batch_size=32, temporal_unit=temporal_unit)
        if encoder_weights:
            self.ts2vec.net.load_state_dict(encoder_weights)
        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_normalization_layer = nn.LayerNorm(input_size)
            self.batch_normalization_layer = self.batch_normalization_layer.double()
            
        if regressor_weights:
            self.ridge = pickle.loads(regressor_weights)
        else:
            self.ridge = MultiOutputRegressor(Ridge(random_state=81800))
    
    def predict(self, x):
        # nomalizin n data
        # if self.batch_norm:
        #     x = self.batch_normalization_layer(x)
        # x: shape (batch_size, sequence_length, input_size)
        encoding = self.ts2vec.encode(x, encoding_window='full_series')
        pred = self.ridge.predict(encoding)
        
        return pred
    
    def save_model(self, path):
        self.hyperparams["encoder_weights"] = self.ts2vec.net.state_dict()
        self.hyperparams["regressor_weights"] = pickle.dumps(self.ridge)
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)
        
        # Find the next available filename
        base_filename = 'ts2vecregressor'
        extension = '.pth'
        counter = 1
        while os.path.exists(os.path.join(path, f"{base_filename}_{counter}{extension}")):
            counter += 1
        
        # Save the model with the numbered filename
        torch.save({
            'model_hyperparams': self.hyperparams,  # like input size, layer sizes, etc
        }, os.path.join(path, f"{base_filename}_{counter}{extension}"))