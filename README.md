# Yet Another Charming Trader - Yacht

## Requirements
* Code tested with `Python 3.7.9`, `tensorflow 2.4.1`, `cuda 11.1` & `cudnn 8.0` on `Windows 10`
* Requirements are installed from the `Pipfile.lock`
( all commands are run at the `Pipfile` folder level).
  
To install your dependencies run:
```shell
pipenv sync
```
To activate the environment:
```shell
pipenv shell
```

## Train
```shell
python train.py --config-file config/crypto.yaml
```

