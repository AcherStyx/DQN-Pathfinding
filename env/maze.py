import gym
import env
import logging
import numpy as np
import cv2

from time import sleep
from gym import spaces

logger = logging.getLogger(__name__)


class MazeEnv(gym.Env):

    def __init__(self):
        self.target = (0, 0)
        self.start = (-5, -5)
        self.map_size = 10

        self.coord_offset = self.map_size

        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(np.array([-self.map_size, self.map_size]),
                                            np.array([-self.map_size, self.map_size]))
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
        x, y = self.state

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

        self.state = np.array((x, y))

        if x == 0 and y == 0:
            done = True
            reward = 10
        elif np.abs(x) >= self.map_size or np.abs(y) >= self.map_size:
            done = True
            reward = -50
        else:
            done = False
            reward = -0.1

        # observation, reward, done
        return self.state, reward, done

    def reset(self):
        self.state = self.start
        try:
            cv2.destroyWindow("env render")
        except cv2.error:
            pass

    def render(self, mode="human", view_size=(400, 400), show=False):
        bg = np.ones((self.map_size * 2 + 1, self.map_size * 2 + 1, 3), dtype=np.float)
        bg[self.coord_offset + 0][self.coord_offset + 0] = self._COLOR_DESTINATION
        assert self.state is not None, "Error: Call reset() first!"
        try:
            bg[self.coord_offset + self.state[0], self.coord_offset + self.state[1]] = self._COLOR_BOT
        except IndexError:
            pass
        bg = cv2.resize(bg, view_size, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("env render", bg)
        cv2.waitKey(1)
        return bg


if __name__ == '__main__':
    env = gym.make('MazeEnv-v0')
    env.reset()
    for _ in range(1000):
        env.render()
        state, reward, done = env.step(action=1)  # take a random action
        sleep(1)
        print(state, reward, done)
    env.close()
