import square_env
from curious_agent import CuriousAgent
import curious_agent as cru
import gym
import numpy as np
import time
import square_env.envs as sqv
import matplotlib.pyplot as plt
from matplotlib import style
import threading
import random
from draw_plots import draw_plots, plot_together, plot_field
from activate_agent import activate_agent
from copy import deepcopy
from random_agent import RandomAgent
from neural_network import NeuralNetwork
import datetime
import stats
from moving_cube import MovingCube

NUM_OF_EPOCHES = 100

PRINT_STATE_PRED = 50

PRINT_TIME_STEP = 500

EPOCH_TIME = 100

NUMBER_OF_AGENTS = 1


def add_avg_dict(src, d, i):
    if i == 0:
        for j in d:
            src[j] = np.array(d[j]) if isinstance(d[j], list) else d[j]
    else:
        for j in src:
            if isinstance(src[j], np.ndarray) and src[j].dtype == 'float':
                src[j] = (float(i) * src[j] + np.array(d[j])) / float(i + 1)


def get_agent_dict(all_agents_dict, index=0):
    d = {}
    for i in all_agents_dict:
        d[i] = all_agents_dict[i][index]
    return d


def join_dict_list(lst):
    d = {}
    for i in lst[0]:
        if isinstance(lst[0][i], list):
            d[i] = np.array(lst[0][i])
            if d[i].dtype == 'float':
                d[i] /= len(lst)
        else:
            d[i] = lst[0][i]
    for i in range(1, len(lst)):
        for j in lst[i]:
            if isinstance(d[j], np.ndarray) and d[j].dtype == 'float':
                d[j] += np.array(lst[i][j]) / len(lst)
    return d


def main():
    agent_dict = []
    errors = []
    wall1 = MovingCube(1)
    print('began running at %s' %  datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
    for i in range(1):

        learner = NeuralNetwork(cru.AGENT_LEARNER_NETWORK_SHAPE, cru.linear_relu, min=-0.1, max=0.1)
        curious_agent = CuriousAgent(0)
        curious_agent.learner = learner
        #sqv.set_global('AGENTS_COUNT', 1)
        d = activate_agent(10, 1000, render=False, print_info=False, reset_env=False, reset_agent=True,
                           agents=[curious_agent, wall1], init_learners=[deepcopy(learner)]*2, get_last_step_avg_error=True,
                           moving_walls_amount=1, moving_wall_start_index=1)

        agent_dict.append(get_agent_dict(d))
        print('finished running #%i at %s' % (i + 1, datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")))
    agent_dict = join_dict_list(agent_dict)
    fig1, ax1 = plot_together(np.arange(len(agent_dict['last_errors'])), [agent_dict['last_errors'],{'label':'curious', 'color':'blue'}],
                              title='Total Errors STD', axis_labels=['epoch', 'last error'])

    plt.show()

    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
