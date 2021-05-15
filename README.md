# Yet Another Charming Trader - Yacht

## Requirements
* Code tested with `Python 3.8`, `pytorch 1.8.1`, `cuda 11.1` & `cudnn 8.0` on `Ubuntu 20.04`
* Requirements are installed from `requirements.txt` within a `conda` environment

* The configuration system is built upon `proto bufs`. If you want to recompile the proto bufs files,
you should install the `protoc` compiler, on your system, from the following [link](https://developers.google.com/protocol-buffers/docs/downloads).
  [Here](https://askubuntu.com/questions/1072683/how-can-i-install-protoc-on-ubuntu-16-04) are the installing steps or read the `README.md`
The `compilation` command is run from the root folder:  
```shell
  protoc -I=. --python_out=. yacht/config/proto/*.proto
  ```
## Train
```shell
python train.py --config_file config/crypto.yaml
```

