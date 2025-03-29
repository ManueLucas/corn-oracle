# corn-oracle
An unfussy (and possibly flexible) corn futures prediction model using curated weather data and ZC (CBOT corn futures) historical pricing data.

## Brought to you by: [Abdullah](https://github.com/AbdullahAswad), [Saad](https://github.com/SaadSheikh02), [and me (Owen)](https://github.com/ManueLucas)
This project was originally made for COMP4107, taught by Matthew Holden at Carleton University.

# WIP
How to train autoregressor:

usage: train_autoregressor.py  

[-h] [--sequence_length SEQUENCE_LENGTH] [--input_size INPUT_SIZE] [--output_size OUTPUT_SIZE] [--hidden_size HIDDEN_SIZE] [--num_layers NUM_LAYERS] [--num_epochs NUM_EPOCHS]
                              [--learning_rate LEARNING_RATE] [--device DEVICE]

optional arguments:

  -h, --help            show this help message and exit
  
  --sequence_length SEQUENCE_LENGTH
  
                        Length of input sequences.
                        
  --hidden_size HIDDEN_SIZE
  
                        Number of features in the hidden state.
                        
  --num_layers NUM_LAYERS
  
                        Number of stacked LSTM layers.
                        
  --num_epochs NUM_EPOCHS
  
                        Number of training epochs.
                        
  --learning_rate LEARNING_RATE
  
                        Learning rate for the optimizer.

  --device DEVICE       
  
                        'cpu' or 'cuda'.
Example: 

python ./train_autoregressor.py --hidden_size 256 --num_layers 2 --num_epochs 5 --dataset combined_data_2000-08-01_to_2025-01-01.csv --device cpu