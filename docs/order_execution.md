# Description
For now our codebase is inspired by the paper [Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860).
If you are interested to see the main concepts I wrote a summary on [Medium](https://medium.com/mlearning-ai/universal-trading-for-order-execution-with-reinforcement-learning-a62d400f2f1a).
We made a few changes to make it more generic & robust:
* The setup can we used on any kind of asset timeseries.
* The actions are taken only at one timestep based on some lagged observations.
* You can use it at any timescale: 1 year, 1 month, 1 day
* We have improved the preprocessing method so the data is stationary.
* We have added a supervised head to help learning.
* We have added as input reference indexes, like S&P500 & Nasdaq100 to help the agent to find the trend.
* We have improved the model to help the agent to pick features from different variables.

A detailed explanation of those improvements can be found at the following [paper]().
<br>Here are some examples of how the agent is performing:
![graph2](../images/graph1_opds.png)
![graph3](../images/graph2_opds.png)
![graph4](../images/graph3_opds.png)

This is how the agent is taking the actions:
![actions](../images/window_logic.jpg)
