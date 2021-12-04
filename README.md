# Yet Another Charming Trader - Yacht
A Deep Reinforcement Learning framework that has as its purpose to aggregate the logic from multiple
papers. By making the system very configurable it would be easier for further researchers to push into the domain
of trading & portfolio management with DRL.

# Install
### Requirements
* Code tested with `Python 3.8`, `pytorch 1.8.1`, `cuda 11.1` & `cudnn 8.0` on `Ubuntu 20.04`
* Requirements are installed from `requirements.txt` within a `conda` environment
* Install torch separately with:
```shell
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

### Config Compiler
* The configuration system is built upon `proto bufs`. If you want to recompile / change the proto bufs files,
you should install the `protoc` compiler on your system:
```shell
sudo apt  install protobuf-compiler
```
* The `compilation` command is running from the root folder:  
```shell
  protoc -I=. --python_out=. yacht/config/proto/*.proto
  ```

### Add Secret Keys
* Create a file called `.env` at the root directory level. If you want to fully use the market APIs and
experiment trackers you should add the secret keys.
* Look at `.env.default` for the supported env vars.

# Run
* All the supported configs can be found at `./yacht/config/configs`.

### Train
```shell
python main.py train --config_file day.config.txt --storage_path ./storage/day
```

### Backtest
```shell
python main.py backtest --config_file day.config.txt --storage_path ./storage/day
```

# Experiment Tracking
### Weights & Biases
* We support wandb for experiment tracking and logging.
* Just at the api key in the `.env` file and in the configuration file you should add the following line:
```shell
meta: {
  experiment_tracker: 'wandb'
}
```

# Hyperparameter Optimization
### Weights & Biases
* Hyperparameter optimization with weights & biases sweeps.
* Weights & biases should work as a simple experiment tracker before using this.
* You can use any other config from `tools/tuning/configs` or generate your own.

```shell
wandb sweep tools/tuning/configs/single_asset_order_execution_crypto.yaml
wandb agent id-given-by-generated-sweep
```

# Data APIs
* Currently we have support for `Binance` & `Yahoo Finance`.
* You should set the `api keys` in the `.env` file for full support.

# Datasets
* S&P 500
* Dow 30
* Nasdaq 100
* Russell 2000

Just set `tickers: ['NASDAQ100']` in the configuration file and all the tickers will be downloaded.
You can also set something like `['NASDQ100', 'RUSSELL2000', 'AAPL']` or any combination you like.
