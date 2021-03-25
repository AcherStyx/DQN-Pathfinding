from gym.envs.registration import register

register(
    id='SimpleTestEnv-v0',
    entry_point='env.simple_test:SimpleTestEnv',
)

register(
    id="MazeEnv-v0",
    entry_point="env.maze:MazeEnv"
)
