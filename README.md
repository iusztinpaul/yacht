# Yet Another Charming Trader - Yacht

# Install
## Requirements
* Code tested with `Python 3.8`, `pytorch 1.8.1`, `cuda 11.1` & `cudnn 8.0` on `Ubuntu 20.04`
* Requirements are installed from `requirements.txt` within a `conda` environment
* Install torch separately with:
```shell
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Config Compiler
* The configuration system is built upon `proto bufs`. If you want to recompile / change the proto bufs files,
you should install the `protoc` compiler on your system:
```shell
sudo apt  install protobuf-compiler
```
* The `compilation` command is running from the root folder:  
```shell
  protoc -I=. --python_out=. yacht/config/proto/*.proto
  ```

## Add Secret Keys
* Create a file called `.env` at the root directory level. If you want to fully use the market APIs and
experiment trackers you should add the secret keys.
* Look at `.env.default` for the supported env vars.

# Run
* All the supported configs can be found at `./yacht/config/configs`.

## Train
```shell
python main.py train --config_file day.config.txt --storage_path ./storage/day
```

## Backtest
```shell
python main.py backtest --config_file day.config.txt --storage_path ./storage/day
```

## Max Possible Profit / Baseline
```shell
python main.py baseline --config_file day.config.txt --storage_path ./storage/day
```

# Experiment Tracking
## Weights & Biases
* We support wandb for experiment tracking and logging.
* Just at the api key in the `.env` file and in the configuration file you should add the following line:
```shell
meta: {
  experiment_tracker: 'wandb'
}
```
