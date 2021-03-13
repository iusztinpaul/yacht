# Yet Another Charming Trader - Yacht

## Requirements
* Code tested with `Python 3.7.9`, `pytorch 1.8.0`, `cuda 11.1` & `cudnn 8.0` on `Windows 10`
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

Finally install `pytorch` ( you should be in the pipenv environment while doing this step):
```shell
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Train
```shell
python train.py --config-file config/crypto.yaml
```

