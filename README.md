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
python main.py train --config-file day.config.txt --storage_path ./storage/day
```
### Resume
```shell
python main.py train --config-file day.config.txt --storage-path ./storage/day --resume-from latest-train
```
### Backtest
```shell
python main.py backtest --config-file day.config.txt --storage-path ./storage/day
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

#### Example of Input Config
```
input: {
    market: 'Yahoo'
    market_mixins: ['TargetPriceMixin']
    dataset: 'DayFrequencyDataset'
    num_assets_per_dataset: 1
    scaler: 'MinMaxScaler'
    scale_on_interval: '1d'
    tickers: ['NASDAQ100']
    fine_tune_tickers: ['AAPL']
    intervals: ['1d']
    features: ['Close', 'Open', 'High', 'Low', 'Volume']
    decision_price_feature: 'TP'
    take_action_at: 'current'
    technical_indicators: ['macd', 'rsi_30']
    start: '1/8/2016'
    end: '15/10/2021'
    period_length: '1M'
    window_size: 30
    render_periods: [
        # For the train period plot a bigger interval, because it is rendered only once.
         {start: '1/1/2018', end: '1/7/2019'},
        # For the validation period be more greedy, because it is rendered a lot of times during training.
         {start: '15/12/2020', end: '15/8/2021'}
    ]
    include_weekends: false
    validation_split_ratio: 0.3
    backtest_split_ratio: 0.0
    embargo_ratio: 0.025
    backtest: {
        run: false
        deterministic: true
        tickers: ['AAPL']
    }
}
```

# Reinforcement Learning Components
## Agents
* PPO

#### Example of Agent Config
```
agent: {
    name: 'StudentPPO'
    is_classic_method: false
    is_teacher: false
    is_student: true
    verbose: true
    policy: {
        name: 'MlpPolicy'
        activation_fn: 'Tanh',
        feature_extractor: {
            name: 'RecurrentFeatureExtractor'
            features_dim: [64,64,128]
            drop_out_p: 0.,
            rnn_layer_type: 'GRU'
        }
        net_arch: {
            shared: [64, 64]
            vf: [32]
            pi: [32]
        }
    }
}
```

## Environments
* SingleAssetEnvironment
* MultiAssetEnvironment
* OrderExecutionEnvironment

## Reward Schemas
#### Trading:
* AssetsPriceChangeRewardSchema

#### Order Execution
* DecisionMakingRewardSchema
* ActionMagnitudeRewardSchema

## Action Schemas
* DiscreteActionScheme
* ContinuousFloatActionSchema
* ContinuousIntegerActionSchema

#### Example of Environment Config
```
environment: {
    name: 'StudentOrderExecutionEnvironment-v0'
    n_envs: 6
    envs_on_different_processes: false
    buy_commission: 0.00
    sell_commission: 0.00
    initial_cash_position: 5000
    reward_schemas: [
    {
        name: 'DecisionMakingRewardSchema',
        reward_scaling: 40
    },
    {
        name: 'ActionMagnitudeRewardSchema',
        reward_scaling: 1.
    }
    ]
    global_reward_scaling: 1.
    action_schema: 'DiscreteActionScheme'
    possibilities: [0, 0.25, 0.5, 0.75, 1]
}
```

#### Example of Train Config
```
train: {
    trainer_name: 'Trainer'
    total_timesteps: 2000000
    fine_tune_total_timesteps: -1
    collecting_n_steps: 2048
    learning_rate: 0.0002
    batch_size: 2048
    n_epochs: 5
    gamma: 1.
    gae_lambda: 1.
    clip_range: 0.3
    vf_clip_range: 0.3
    entropy_coefficient: 0.6
    vf_coefficient: 1.
    max_grad_norm: 100
    use_sde: false
    sde_sample_freq: -1
    learning_rate_scheduler: 'ConstantSchedule'
}
```

## Detailed Documentation
* [Order Execution](docs/order_execution.md)
