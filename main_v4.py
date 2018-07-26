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
    random_dict = []
    random_agent = RandomAgent(0)
    wall1, wall2 = MovingCube(1), MovingCube(2)
    rnd_ag_list = [random_agent, wall1]
    print('began running at %s' %  datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S"))
    for i in range(5):


        d = activate_agent(10000, render=True, print_info=False, reset_env=True, number_of_agents=2, get_avg_errors=False,
                           get_values_field=False, number_of_error_agents=1)
        agent_dict.append(get_agent_dict(d))

        print('finished running #%i at %s' % (i + 1, datetime.datetime.now().strftime("%a, %d %B %Y %H:%M:%S")))

    means_curious = []
    for i in agent_dict:
        means_curious.append(i['total_errors'])
    std_agent = np.array(means_curious).std(axis=0)

    means_random = []
    for i in random_dict:
        means_random.append(i['total_errors'])
    std_random = np.array(means_random).std(axis=0)


    agent_dict = join_dict_list(agent_dict)
    #draw_plots(agent_dict)
    random_dict = join_dict_list(random_dict)
    #draw_plots(random_dict)

    #fig, ax ,q = plot_field(*agent_dict['fields'], title='Agent Value Field', color=agent_dict['fields_colors'])

    errors_rate_curious = agent_dict['total_errors']
    errors_rate_random = random_dict['total_errors']


    fig1, ax1 = plot_together(agent_dict['timesteps'], [errors_rate_curious, {'label':'curious', 'color':'blue'}],
                  [errors_rate_random, {'label':'random', 'color':'red'}], title='Total Errors STD',
                  std=[std_agent, std_random], axis_labels=['steps', 'total error'])

    fig2, ax2 = plot_together(agent_dict['timesteps'], [errors_rate_curious, {'label': 'curious', 'color': 'blue'}],
                  [errors_rate_random, {'label': 'random', 'color': 'red'}], title='Total Errors Means',
                  means=[means_curious, means_random], axis_labels=['steps', 'total error'])

    fig3, ax3 = plot_together(agent_dict['timesteps'][:-1], [stats.derivative(errors_rate_curious), {'label': 'curious', 'color': 'blue'}],
                  [stats.derivative(errors_rate_random), {'label': 'random', 'color': 'red'}], title='Total Errors Derivative',
                axis_labels=['steps', 'total error'])

    fig1.savefig('./plots/std.png')
    fig2.savefig('./plots/means.png')
    plt.show()

    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
