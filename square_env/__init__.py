from gym.envs.registration import register

register(
    id='square-v0',
    entry_point='square_env.envs:SquareEnv',
)