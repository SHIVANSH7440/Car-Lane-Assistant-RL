
import gym
from gym import spaces
import numpy as np

class LaneKeepingEnv(gym.Env):
    """
    Custom Lane Keeping Environment for Reinforcement Learning
    """

    def __init__(self):
        super(LaneKeepingEnv, self).__init__()

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        self.reset()

    def reset(self):
        self.car_x = 0.0
        self.velocity = 0.5
        self.steps = 0
        return self._get_state()

    def step(self, action):
        self.steps += 1

        if action == 0:
            self.car_x -= 0.05
        elif action == 2:
            self.car_x += 0.05

        distance = abs(self.car_x)

        if distance < 0.2:
            reward = 1.0
        elif distance < 0.5:
            reward = -0.1
        else:
            reward = -10.0

        done = distance > 1.0 or self.steps >= 200
        return self._get_state(), reward, done, {}

    def _get_state(self):
        return np.array([self.car_x, self.velocity, abs(self.car_x)], dtype=np.float32)
