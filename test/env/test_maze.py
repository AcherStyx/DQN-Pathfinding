import cv2
import gym

from env.maze import *

import unittest


class TestMaze(unittest.TestCase):

    def test_visualize(self):
        inst = MazeEnv()
        inst.reset()
        end = False
        print("direction | rewards")
        while not end:
            _action = random.randint(0, 3)
            _, _reward, end, _ = inst.step(_action)
            if _action == 0:
                direction = "  up   "
            elif _action == 1:
                direction = "  right"
            elif _action == 2:
                direction = "  down "
            elif _action == 3:
                direction = "  left "
            else:
                raise ValueError
            print(direction, _reward)
            img = inst.render()
            cv2.imshow("visual", img)
            cv2.waitKey(0)

    def test_gym_make(self):
        env = gym.make("MazeEnv-v0")
        env.reset()
        img = env.render()
        cv2.imshow("make", img)
        cv2.waitKey(0)
