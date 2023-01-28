import time
import numpy as np

from .PendulumDomainRandomization import PendulumDomainRandomization


class Pendulum(object):
    def __init__(self, userDefinedSettings, domain_range=None):
        self.userDefinedSettings = userDefinedSettings
        self.env = PendulumDomainRandomization(userDefinedSettings, domain_range)
        self.DOMAIN_PARAMETER_DIM = self.env.get_domain_parameter_dim()
        self.RENDER_INTERVAL = 0.05  # [s]
        self.MAX_EPISODE_LENGTH = 150
        self.ACTION_MAPPING_FLAG = True

        self.STATE_DIM = self.env.observation_space.shape[0]
        self.ACTION_DIM = self.env.action_space.shape[0]

        self.domainInfo = self.env.domainInfo

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action, get_task_achievement=False):
        if self.ACTION_MAPPING_FLAG:
            action = self.mapping_action(action)
        next_state, reward, done, domain_parameter, task_achievement = self.env.step(action)

        if get_task_achievement:
            return next_state, reward, done, domain_parameter, task_achievement
        else:
            return next_state, reward, done, domain_parameter

    def random_action_sample(self):
        action = self.env.action_space.sample()
        if self.ACTION_MAPPING_FLAG:
            low = self.env.action_space.low
            high = self.env.action_space.high
            action = 2 * (action - low) / (high - low) - 1
        return action

    def render(self):
        self.env.render()
        time.sleep(self.RENDER_INTERVAL)

    def mapping_action_discrete2continues(self, action):
        return self.actions[action]

    def mapping_action(self, action):
        assert (action >= -1) and (action <= 1), 'expected actions are \"-1 to +1\". input actions are {}'.format(action)
        low = self.env.action_space.low
        high = self.env.action_space.high
        action = low + (action + 1.0) * 0.5 * (high - low)  # (-1,1) -> (low,high)
        action = np.clip(action, low, high)  # (-X,+Y) -> (low,high)
        return action

    def __del__(self):
        self.env.close()

    def user_direct_set_domain_parameters(self, domain_info, type='set_split2'):
        self.env.domainInfo.set_parameters(domain_info, type=type)
