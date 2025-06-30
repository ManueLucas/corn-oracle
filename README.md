# corn-oracle
An unfussy (and possibly flexible) corn futures prediction model using curated weather data and ZC (CBOT corn futures) historical pricing data.

## Brought to you by: [Abdullah](https://github.com/AbdullahAswad), [Saad](https://github.com/SaadSheikh02), [and me (Owen)](https://github.com/ManueLucas)
This project was originally made for COMP4107, taught by Matthew Holden at Carleton University.

[📄 Read the term paper](./OracleReport.pdf)

We use YAML files to configure training hyperparameters, feature selection, model variants, and test splits. RNN and TS2Vec models have different hyperparameters, so take care to check the examples we've left.
There are 2 folders containing yaml files for the models featured in the [📄 report](./OracleReport.pdf) in ./config: "rnn" and "ts2vec", for their respective architectures.

How to train TS2Vec: python train_validate_ts2vec.py --config_path [CONFIG_PATH] --print_predictions (flag)
Example: python train_validate_ts2vec.py --config_path ./configs/ts2vec/kernel_encoding_256.yaml

How to train LSTM: python train_validate_rnn.py --config_path [CONFIG_PATH] --print_predictions (flag)
Example: python train_validate_rnn.py --config_path ./configs/rnn/example.yaml

Yes, setup is messy and redundant in a lot of places, and no, I won't be fixing it. If there is a change, this entire repo will be bulldozed for it.
