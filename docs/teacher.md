# Description
In our case the teacher is used as in [Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860).
It is used as an oracle that has access to all the epoch data. In this way we can generate ideal actions that can be further
used as GT for a student agent. This will help & stabilize learning.

# Usage
Train teacher:
```shell
python main.py train --config-file order_execution/all/single_asset_all_opdt.config.txt --storage_path ./storage/opdt --market-storage-dir ./storage
```
Export the actions:
```shell
export_actions --config-file-name order_execution/all/single_asset_all_opdt_export.config.txt --storage-dir storage/single_asset_all_opdt --market-storage-dir ./storage
```
Train the student with the exported actions:
```shell
python main.py train --config-file order_execution/all/single_asset_all_opd.config.txt --storage_path ./storage/opd --market-storage-dir ./storage
```
