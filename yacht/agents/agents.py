from stable_baselines3 import PPO as SB3PPO
from stable_baselines3 import SAC as SB3SAC


class PPO(SB3PPO):
    def train(self) -> None:
        super().train()

        self.logger.dump()


class SAC(SB3SAC):
    pass
