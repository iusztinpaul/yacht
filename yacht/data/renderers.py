class BaseRenderer:
    def render(self):
        raise NotImplementedError()


class PriceRenderer(BaseRenderer):
    def render(self):
        pass


class RewardRenderer(BaseRenderer):
    def render(self):
        pass


class ActionRenderer(BaseRenderer):
    def render(self):
        pass


class Renderer(BaseRenderer):
    price_renderer = PriceRenderer
    reward_renderer = RewardRenderer
    action_renderer = ActionRenderer

    def render(self):
        pass
