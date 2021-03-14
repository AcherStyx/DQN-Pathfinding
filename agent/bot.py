import tensorflow as tf
import tf_agents
import env
import time

from tf_agents.environments import suite_gym


def main():
    maze_env = suite_gym.load("MazeEnv-v0")
    maze_env.reset()
    maze_env.render()

    print(maze_env.time_step_spec().observation)


if __name__ == '__main__':
    main()
