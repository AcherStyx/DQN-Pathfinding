import random

import gym
import env
import logging
import numpy as np
import cv2

from time import sleep
from gym import spaces

logger = logging.getLogger(__name__)


class SimpleTestEnv(gym.Env):

    def __init__(self):
        """
        action:
        0-> not move
        1-> up
        2-> right
        3-> down
        4-> left
        observation:
        (x,y)-> current position
        :rtype: gym.Env
        """
        self.target = (0, 0)
        # self.start = (-5, -5)
        self.map_size = 10

        self.step_count = 0
        self.max_step = 20

        self.coord_offset = self.map_size

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(np.array([-self.map_size, -self.map_size, 0]),
                                            np.array([self.map_size, self.map_size, self.max_step]))
        self.state = None  # need reset before use

        # color BGR
        self._COLOR_DESTINATION = (0, 255, 0)
        self._COLOR_BOT = (0, 0, 255)

    def step(self, action):
        # action:
        #   1    y^
        # 4 0 2  x->
        #   3
        assert self.action_space.contains(action)
        x, y, step = self.state

        if action == 0:  # not move
            pass
        elif action == 1:
            y += 1
        elif action == 2:
            x += 1
        elif action == 3:
            y -= 1
        elif action == 4:
            x -= 1

        self.step_count += 1
        self.state = np.array((x, y, self.step_count))

        if self.step_count > self.max_step:
            done = True
            reward = -30
        elif x == 0 and y == 0:
            done = True
            reward = 100
        elif np.abs(x) > self.map_size or np.abs(y) > self.map_size:
            done = True
            reward = -50
        else:
            done = False
            reward = -0.1

        # observation, reward, done, info
        return self.state, reward, done, None

    def reset(self):
        self.step_count = 0
        self.state = (random.randint(-self.map_size, self.map_size),
                      random.randint(-self.map_size, self.map_size),
                      self.step_count)
        try:
            cv2.destroyWindow("env render")
        except cv2.error:
            pass
        return self.state

    def render(self, mode="human", view_size=(400, 400), show=True):
        bg = np.ones((self.map_size * 2 + 1, self.map_size * 2 + 1, 3), dtype=np.float)
        bg[self.coord_offset + 0][self.coord_offset + 0] = self._COLOR_DESTINATION
        assert self.state is not None, "Error: Call reset() first!"
        try:
            bg[self.coord_offset + self.state[0], self.coord_offset + self.state[1]] = self._COLOR_BOT
        except IndexError:
            pass
        bg = cv2.resize(bg, view_size, interpolation=cv2.INTER_NEAREST)
        if show:
            cv2.imshow("env render", bg)
            cv2.waitKey(1)
        return bg


if __name__ == '__main__':
    env = gym.make('SimpleTestEnv-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        state, reward, done, info = env.step(action=1)  # take a random action
        sleep(1)
        print(state, reward, done)
    env.close()
