from agent.path_finder import *

from unittest import TestCase
import env


class TestPathFinder(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestPathFinder, self).__init__(*args, **kwargs)
        logging.basicConfig(level=logging.DEBUG)

    def test_on_simple_test(self):
        train_env = suite_gym.load("SimpleTestEnv-v0")
        eval_env = suite_gym.load("SimpleTestEnv-v0")
        step_spec = train_env.reset()

        logger.debug("Reset return time_step_spec: %s", step_spec)
        logger.debug("Observation: %s", train_env.time_step_spec().observation)
        logger.debug("Reward: %s", train_env.time_step_spec().reward)
        logger.debug("Action: %s", train_env.action_spec())

        inst = PathFinder(cfg=PathFinder.Config(train_env=train_env, eval_env=eval_env,
                                                batch_size=64, log_dir="../../log"))

        print("reward baseline:", inst._metric(100))
        inst.train(20000)
        while True:
            inst.visualize_demo()

    def test_on_maze(self):
        train_env = suite_gym.load("MazeEnv-v0")
        eval_env = suite_gym.load("MazeEnv-v0")
        step_spec = train_env.reset()

        logger.debug("Reset return time_step_spec: %s", step_spec)
        logger.debug("Observation: %s", train_env.time_step_spec().observation)
        logger.debug("Reward: %s", train_env.time_step_spec().reward)
        logger.debug("Action: %s", train_env.action_spec())

        inst = PathFinder(cfg=PathFinder.Config(train_env=train_env, eval_env=eval_env,
                                                batch_size=64, log_dir="../../log"))

        print("reward baseline:", inst._metric(100))
        inst.train(20000)
        while True:
            inst.visualize_demo()
