import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

from .PendulumDomainInfo import PendulumDomainInfo


class PendulumDomainRandomization(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, userDefinedSettings, domain_range=None):
        self.userDefinedSettings = userDefinedSettings
        self.domainInfo = PendulumDomainInfo(userDefinedSettings, domain_range)
        self.max_speed = 8  # 8
        self.max_torque = 2.  # 2.
        self.viewer = None
        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,))
        self.observation_space = spaces.Box(low=-high, high=high)
        self._seed()

    def set_domain_parameter(self, domain_info=None):
        if domain_info is not None:
            self.domainInfo.set_parameters(set_info=domain_info)
        else:
            self.domainInfo.set_parameters()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        th, thdot = self.state  # th := theta

        domain_parameter = self.domainInfo.get_domain_parameters()
        dt, g, m, l, torque_weight, torque_bias = domain_parameter

        u = u * torque_weight + torque_bias  # for using domain randomization

        if self.domainInfo.torque_limit:
            u = np.clip(u, -self.max_torque, self.max_torque)[0]
        else:
            u = u[0]
        self.last_u = u  # for rendering
        reward = self.calc_reward(th, thdot, u)

        newthdot = thdot + (-3 * g / (2 * l) * np.sin(th + np.pi) + 3. / (m * l**2) * u) * dt
        newth = th + newthdot * dt
        if self.domainInfo.velocity_limit:
            newthdot = np.clip(newthdot, -self.max_speed, self.max_speed)

        self.state = np.array([newth, newthdot])

        observation = self._get_obs()
        done = False
        self.step_num += 1
        return observation, reward, done, domain_parameter, False

    def calc_reward(self, th, thdot, u):
        costs = angle_normalize(th)**2 + 0.1 * abs(thdot) + 0.001 * abs(u)
        return -costs

    def get_domain_parameter_dim(self):
        return self.domainInfo.get_domain_parameters().shape[0]

    def reset(self):
        self.step_num = 0
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-2.2, 2.2, -2.2, 2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0, 0, 0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi / 2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u / 2, np.abs(self.last_u) / 2)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)
