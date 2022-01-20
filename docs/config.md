# Main Config Structure
We use `Google Protobuf` to model the config system. 
Used examples can be seen at: `yacht\config\configs`
This is the big picture of the config:
```
input: {
    ...
}
environment: {
    ...
}
agent: {
    ...
}
train: {
   ...
}
meta: {
   ...
}
```

## Input Config
```
input: {
    market: 'Yahoo'  # Data Source
    market_mixins: ['TargetPriceMixin', 'FracDiffMixin']  # Mixin that preprocesses the data when is cached.
    dataset: 'DayFrequencyDataset'  # The dataset that models a single asset. 
                                    # There is an aggregator dataset that is used for multiple assets.
    num_assets_per_dataset: 1  # How many assets there will be in a dataset.
    scaler: 'MinMaxScaler'  # The scaler applied on loading.
    scale_on_interval: '1d'  # On what interval to find the scaling paramters.
    tickers: ['NASDAQ100', 'S&P500', 'DOW30']   # On what tickers to train.
    attached_tickers: ['QQQ', 'SPY']  # Extra tickers that will be attached to EVERY dataset.
    fine_tune_tickers: ['NASDAQ100']  # Tickers to fine-tune on.
    intervals: ['1d']  # What intervals/frequency to use.
    features: ['CloseFracDiff', 'OpenFracDiff', 'HighFracDiff', 'LowFracDiff', 'VolumeFracDiff']  # The features to use while training.
    decision_price_feature: 'TP'  # On what features to compute all the rewards & metrics.
    take_action_at: 'next'  # When to take the action. We support only: current & next.
    technical_indicators: ['macdFracDiff', 'rsi_30']  # Technical indicators to compute & use.
    start: '1/8/2016'  # The start of the whole dataset.
    end: '15/10/2021'  # The end of the whole dataset.
    period_length: '1M'  # The length of an episode.
    window_size: 5  # Number of lagged observations to look on.
    render_periods: [  # Render subsets. Rendering is an expensive computation. If is it empty no rendering will be done.
         {start: '1/1/2018', end: '1/7/2019'},
         {start: '15/12/2020', end: '15/8/2021'}
    ]
    render_tickers: ['AAPL']  # Render tickers. Rendering is an expensive computation. If is it empty no rendering will be done.
    include_weekends: false  # If the assets are traded during workdays or not.
    validation_split_ratio: 0.2  # How much of the data to add to train.
    backtest_split_ratio: 0.1  # How much of the data to add to testing.
    embargo_ratio: 0.025  # Embargo ratio between splits.
    backtest: {
        run: false  # Flag to backtest on the testing set or not.
        deterministic: true  # Run the backtesting in a deterministic way or not.
        tickers: ['NASDAQ100']  # Backtesting/Validation tickers.
    }
}
```

## Environment Config
```
environment: {
    name: 'OrderExecutionEnvironment-v0'  # What environment to use.
    n_envs: 6  # The number of environments.
    envs_on_different_processes: false  # Flag to run environments on differenct processes.
    buy_commission: 0.00  # Buy commission.
    sell_commission: 0.00  # Sell comission.
    initial_cash_position: 5000  # The cash position that the agent is starting with.
    reward_schemas: [  # List of reward schemas.
    {
        name: 'DecisionMakingRewardSchema',
        reward_scaling: 1.
    },
    {
        name: 'ActionMagnitudeRewardSchema',
        reward_scaling: 0.025
    }
    ]
    global_reward_scaling: 1.  # Reward scaling that is applied after all the rewards are computed.
    action_schema: 'DiscreteActionScheme'  # Action schema.
    possibilities: [0, 0.25, 0.5, 0.75, 1]  # Action schema possibilities ( if it is discrete)
    action_scaling_factor: 1  # Action schema scaling factor ( if it continous)
}
```

## Agent Config
A lot of parameters coincide with the ones from `stable-baselines3`. Refer to their
[documentation](https://stable-baselines3.readthedocs.io/en/master/) for a more detailed reading.
```
agent: {
    name: 'SupervisedPPO'  # The agent that will be used.
    is_classic_method: false  # Flag if it is a classic method or not.
    is_teacher: false  # Flag if it is used as a teacher or not.
    is_student: false  # Flag if it is used as a student or not.
    verbose: true
    policy: { 
        name: 'MlpPolicy'  # The name of the policy that is used.
        activation_fn: 'Tanh',
        feature_extractor: {
            name: 'DayRecurrentFeatureExtractor'  # The name of the feature extractor that the policy uses.
            features_dim: [64,64,128]
            drop_out_p: 0.,
            rnn_layer_type: 'GRU'
        }
        net_arch: {  # Network that will be called on the outputs of the FeatureExtractor.
            shared: [64, 64]
            vf: [32]
            pi: [32]
        }
    }
}
```

## Training Config
A lot of parameters coincide with the ones from `stable-baselines3`. Refer to their
[documentation](https://stable-baselines3.readthedocs.io/en/master/) for a more detailed reading.
```
train: {
    trainer_name: 'Trainer'  # The trainer to be used.
    total_timesteps: 3000000
    fine_tune_total_timesteps: -1  # Fine tune number of timesteps. If it is `-1` fine tuning is stopped.
    collecting_n_steps: 2048
    learning_rate: 0.0002
    batch_size: 2048
    n_epochs: 5
    gamma: 1.
    gae_lambda: 1.
    clip_range: 0.3
    vf_clip_range: 0.3
    entropy_coefficient: 0.15
    vf_coefficient: 1.
    max_grad_norm: 100
    use_sde: false
    sde_sample_freq: -1
    learning_rate_scheduler: 'ConstantSchedule'
    supervised_coef: 5.  # Coeficient used by the supervised head of the agent.
}
```


##  Meta Config
```
meta: {
    log_frequency_steps: 5000  # Frequency to log state during training.
    metrics_to_log: ['PA', 'GLR', 'cash_used_on_last_tick']  # What metrics to log.
    metrics_to_save_best_on: ['PA', 'GLR']  # On what metrics to save the best agent.
    metrics_to_load_best_on: ['PA', 'GLR']  # On what metrics to resume/backtest the agent.
    plateau_max_n_steps: -1  # Number of validation steps until the training is stopped if the metrics did not improve.
                             # If it is `-1` the logic is stopped. 
    device: 'gpu'  # Either gpu or cpu
    experiment_tracker: 'wandb'  # Experiment tracker to be used. If it is "''" it is stopped.
}
```
