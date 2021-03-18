import tensorflow as tf
import tf_agents
import env
import time

from tf_agents.environments import suite_gym, tf_py_environment
from tf_agents.specs import tensor_spec
from tf_agents.networks import sequential

class PathFinder:

    def build_dqn(self):


def main():
    simple_py_env = suite_gym.load("SimpleTestEnv-v0")
    simple_py_env.reset()
    simple_py_env.render()

    print(simple_py_env.time_step_spec().observation)
    print(simple_py_env.time_step_spec().reward)
    print(simple_py_env.action_spec())

    train_env = tf_py_environment.TFPyEnvironment(simple_py_env)




if __name__ == '__main__':
    main()
