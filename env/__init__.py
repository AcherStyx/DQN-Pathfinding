from gym.envs.registration import register

register(
    id='MazeEnv-v0',
    entry_point='env.maze:MazeEnv',
)
