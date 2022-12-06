# Description
In our case the teacher-student setup is used as in [Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860).
It is used as an oracle that has access to all the epoch data. In this way we can generate ideal actions that can be further
used as GT for a student agent. This will help & stabilize learning.

# Usage
Train teacher:
```shell
python main.py train --config-file order_execution/all/single_asset_all_universal_teacher.config.txt --storage_path ./storage/universal_teacher --market-storage-dir ./storage
```
Export the actions:
```shell
export_actions --config-file-name order_execution/all/single_asset_all_universal_teacher_export.config.txt --storage-dir ./storage/universal_teacher --market-storage-dir ./storage
```
Train the student with the exported actions:
```shell
python main.py train --config-file order_execution/all/single_asset_all_universal_distillation.config.txt --storage_path ./storage/universal_distillation --market-storage-dir ./storage
```
