import numpy as np
import square_env
import gym
import curious_agent as cru
from curious_agent import ALL_ACTIONS
import square_env.envs as sqv
from copy import deepcopy

ALL_DIRECTIONS = np.array([[0, 1],[1, 0],[-1, 0], [0, -1]])


# def train_and_average_errors(agent, epoch_time=1000, render=False, print_info=False):
#     env = gym.make('square-v0')
#     agent.reset_network()
#     activate_agent(epoch_time, reset_agent=False, reset_env=False, env=env, render=render, print_info=print_info)
#     cost_avg = 0.0
#     for x in range(sqv.RECT_WIDTH):
#         for y in range(sqv.RECT_HEIGHT):
#             for direction in ALL_DIRECTIONS:
#                 for action in ALL_ACTIONS:
#                     old_state = env._get_all_observations()[agent.index]
#                     env.agents[agent.index]['loc'] = np.array([x, y])
#                     env.agents[agent.index]['dir'] = np.copy(direction)
#                     state, _, _, _ = env.step(action=action, index=agent.index)
#                     cost_avg += agent.learner.cost(np.array([np.concatenate((old_state, action))]),
#                                                    np.array([state])) / float(sqv.RECT_WIDTH*sqv.RECT_HEIGHT*12)
#     return cost_avg


def average_errors_on_trained_agent(agent, env):
    env = deepcopy(env)
    cost_avg = 0.0
    for x in range(sqv.RECT_WIDTH):
        for y in range(sqv.RECT_HEIGHT):
            for direction in ALL_DIRECTIONS:
                for action in ALL_ACTIONS:
                    old_state = env._get_all_observations()[agent.index]
                    env.agents[agent.index]['loc'] = np.array([x, y])
                    env.agents[agent.index]['dir'] = np.copy(direction)
                    state, _, _, _ = env.step(action=np.array([np.amax(action)]), index=agent.index)
                    cost_avg += agent.learner.cost(np.array([np.concatenate((old_state, action))]),
                                                   np.array([state])) / float(sqv.RECT_WIDTH*sqv.RECT_HEIGHT*12)
    return cost_avg
