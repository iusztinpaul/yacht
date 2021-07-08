# Yet Another Charming Trader - Yacht

# Install
## Requirements
* Code tested with `Python 3.8`, `pytorch 1.8.1`, `cuda 11.1` & `cudnn 8.0` on `Ubuntu 20.04`
* Requirements are installed from `requirements.txt` within a `conda` environment
* Install torch separately with:
```shell
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Config compiler
* The configuration system is built upon `proto bufs`. If you want to recompile / change the proto bufs files,
you should install the `protoc` compiler on your system:
```shell
sudo apt  install protobuf-compiler
```
* The `compilation` command is running from the root folder:  
```shell
  protoc -I=. --python_out=. yacht/config/proto/*.proto
  ```

# Run
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
