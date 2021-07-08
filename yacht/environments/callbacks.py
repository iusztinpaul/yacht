import logging

import wandb
from stable_baselines3.common.callbacks import BaseCallback


logger = logging.getLogger(__file__)


class LoggerCallback(BaseCallback):
    def __init__(
            self,
            collect_n_times: int,
            collecting_n_steps: int,
            verbose: int = 0
    ):
        super().__init__(verbose)

        self.collect_n_times = collect_n_times
        self.collecting_n_steps = collecting_n_steps

    def _on_step(self) -> bool:
        current_episode = self.num_timesteps / self.collecting_n_steps
        if current_episode == int(current_episode):
            logger.info(f'Finished training for collect_n_time: [{current_episode}/{self.collect_n_times}].')

        return True


class WandBCallback(BaseCallback):
    def _on_step(self) -> bool:
        info = self.locals['infos'][-1]
        is_done = info.pop('done')
        episode_metrics = info.pop('episode_metrics', False)
        episode_data = info.pop('episode', False)

        info_to_log = dict()

        info_to_log['total_value'] = info['total_value']
        info_to_log['num_longs'] = info['num_longs']
        info_to_log['num_shorts'] = info['num_shorts']
        info_to_log['num_holds'] = info['num_holds']
        info_to_log['profit_hits'] = info['profit_hits']
        info_to_log['loss_misses'] = info['loss_misses']
        info_to_log['hit_ratio'] = info['hit_ratio']

        if is_done and episode_metrics:
            # TODO: Log more metrics after we understand them.
            info_to_log['episode_metrics'] = {
                'Annual return': episode_metrics['Annual return'],
                'Cumulative returns': episode_metrics['Cumulative returns'],
                'Annual volatility': episode_metrics['Annual volatility'],
                'Sharpe ratio': episode_metrics['Sharpe ratio']
            }

            # Translate the keys for easier understanding
            info_to_log['episode'] = {
                'reward': episode_data['r'],
                'length': episode_data['l'],
                'seconds': episode_data['t']
            }

        wandb.log({
            'train': info_to_log
        })

        return True
