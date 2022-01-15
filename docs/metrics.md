# Description
We use the `pyfolio-reloaded` package to compute standard metrics like:
* sharp ratio
* annual return
* alpha
* beta

For a more detailed explanation refer to their [page](https://pypi.org/project/pyfolio-reloaded/). 
They support a lot of metrics.
<br> The custom metrics implemented in this repository are the following:
* **PA** = Price Advantage
* **GLR** = Gain Loss Ratio
* **AD** = Action Distance
* **ADS** = Action Distance from the Start
* **ADE** = Action Distance from the End
* **cash_used_on_last_tick** = The cash that was used at the end of an episode
* **T** = Tactics, which is just `T = ADS / cash_used_on_last_tick`
* **num_actions** = The number of the actions the agent has taken

**NOTE:** `PA` & `GLR` are used for the `Order Execution Task`. For a more detailed explanation
please refer to the following paper: [Universal Trading for Order Execution with Oracle Policy Distillation](https://arxiv.org/abs/2103.10860).