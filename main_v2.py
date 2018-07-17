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
from draw_plots import draw_plots, plot_together
from activate_agent import activate_agent
from copy import deepcopy
from random_agent import RandomAgent
from neural_network import NeuralNetwork


NUM_OF_EPOCHES = 100

PRINT_STATE_PRED = 50

PRINT_TIME_STEP = 500

EPOCH_TIME = 100

NUMBER_OF_AGENTS = 1


def get_agent_dict(all_agents_dict, index=0):
    d = {}
    for i in all_agents_dict:
        d[i] = all_agents_dict[i][index]
    return d


def join_dict_list(lst):
    d = {}
    for i in lst[0]:
        d[i] = np.array(lst[0][i]) if isinstance(i, list) else lst[0][i]
    for i in xrange(1, len(lst)):
        for j in lst[i]:
            if isinstance(d[j], np.ndarray):
                d[j] += np.array(lst[i][j])
    for i in d:
        if isinstance(d[i],np.ndarray):
            d[i] /= float(len(lst))
    return d


def main():
    agent_dict = []
    random_dict = []

    for i in xrange(500):
        learner = NeuralNetwork(cru.AGENT_LEARNER_NETWORK_SHAPE, cru.relu)
        curious_agent = CuriousAgent(0)
        curious_agent.learner = deepcopy(learner)
        activate_agent(100, 100, render=False, print_info=False, reset_env=True, agents=[curious_agent])

        d = activate_agent(100, render=False, print_info=False, reset_env=True, agents=[curious_agent])
        agent_dict.append(get_agent_dict(d))

        random_agent = RandomAgent(0)
        random_agent.learner = deepcopy(learner)
        d = activate_agent(100, render=False, print_info=False, reset_env=True, agents=[random_agent])

        random_dict.append(get_agent_dict(d))

        print "finished running #%i"%i

    draw_plots(join_dict_list(agent_dict))
    draw_plots(join_dict_list(random_dict))


    from IPython import embed
    embed()


if __name__ == "__main__":
    main()