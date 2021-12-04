# Description
1. Reproduction of [Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860). 
In this codebase the code is more generic. You can choose from different assets to run tests on. Also, you can set up 
your environment for different frequencies as input (eg. 1m, 1h, 1d), reward schemas & action schemas. I played around with
the agent feature extractor to extrapolate it to work on a `window` logic & at different time scales. The original implementation
was working only at the `day level`. My goal was to make a general framework for order execution.


## 1. Universal Trading for Order Execution with Oracle Policy Distillation