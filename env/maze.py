import random

import cv2
import gym
import numpy as np


class MazeEnv(gym.Env):
    class Config:
        maze_size = (10, 10)
        step_max = 50

        road = 0
        bot = 1
        wall = -1
        life = 2
        end = 3
        color = {-1: [0, 0, 0],
                 0: [255, 255, 255],
                 1: [0, 255, 0],
                 2: [255, 0, 0],
                 3: [0, 0, 255]}

    def __init__(self):

        self.cfg = MazeEnv.Config

        self._step_count = 0
        self._stats = None
        self._end = None
        self._bot = None

        self.action_space = gym.spaces.Discrete(4)
        # 0-> up    1-> right   2-> down    3-> left
        self.observation_space = gym.spaces.Box(low=np.ones(self.cfg.maze_size[0] * self.cfg.maze_size[1]) * -1,
                                                high=np.ones(self.cfg.maze_size[0] * self.cfg.maze_size[1]) * 3,
                                                dtype=np.int)

    def reset(self):
        try:
            cv2.destroyWindow("MazeEnv")
        except cv2.error:
            pass

        seed = random.randint(0, 10000000)

        random.seed(seed)

        maze = np.random.rand(*self.cfg.maze_size)
        maze = (maze > 0.8).astype(np.int8) * -1
        self._step_count = 0

        while True:
            self._bot = (random.randint(0, self.cfg.maze_size[0] - 1),
                         random.randint(0, self.cfg.maze_size[1] - 1))
            self._end = (random.randint(0, self.cfg.maze_size[0] - 1),
                         random.randint(0, self.cfg.maze_size[1] - 1))
            if self._end != self._bot:
                break

        maze[self._bot[1], self._bot[0]] = 1
        maze[self._end] = 3

        # print(maze)
        self._stats = maze.flatten()
        return self._stats

    def render(self, mode='human'):
        maze = np.reshape(self._stats, self.cfg.maze_size)
        maze.tolist()
        maze = np.array([[self.cfg.color[i] for i in ii] for ii in maze], dtype=np.uint8)
        cv2.imshow("MazeEnv", maze)
        cv2.waitKey(1)
        return maze

    def step(self, action):
        assert self.action_space.contains(action)
        maze = np.reshape(self._stats, self.cfg.maze_size)

        x, y = self._bot
        maze[y, x] = 0
        if action == 0:
            y -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y += 1
        elif action == 3:
            x -= 1

        self._step_count += 1
        if self._step_count > self.cfg.step_max:
            reward = -20
            done = True
            move = False
        elif (not 0 <= x < self.cfg.maze_size[0]) or (not 0 <= y < self.cfg.maze_size[1]):
            reward = -10
            done = False
            move = False
        elif maze[y, x] == -1:
            reward = -10
            done = False
            move = False
        elif maze[y, x] == 0:
            reward = -0.1
            done = False
            move = True
        elif maze[y, x] == 2:
            reward = 10
            done = False
            move = True
            maze[y, x] = 0
        elif maze[y, x] == 3:
            reward = 100
            done = True
            move = True
        else:
            raise ValueError

        if move:
            maze[y, x] = 1
            self._bot = (x, y)
        else:
            maze[self._bot[1], self._bot[0]] = 1

        self._stats = maze.flatten()

        return self._stats, reward, done, None
