import square_env
from curious_agent import CuriousAgent
import gym
import numpy as np
import time
import square_env.envs as sqv
import matplotlib.pyplot as plt
from matplotlib import style
import threading
import random
from draw_plots import draw_plots
from activate_agent import activate_agent


NUM_OF_EPOCHES = 20

PRINT_STATE_PRED = 50

PRINT_TIME_STEP = 500

EPOCH_TIME = 20

NUMBER_OF_AGENTS = 1


def get_agent_dict(all_agents_dict, index):
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

    d_nonreset = []
    d_reset = []
    d_nontrained = []

    for i in xrange(500):
        d = activate_agent(100, 20, render=False)
        reset_trained_agent = d['agents']
        reset_trained_agent[0].reset_network()


        d = activate_agent(100, 20, reset_agent=False, render=False)
        nonreset_trained_agent = d['agents']
        nonreset_trained_agent[0].reset_network()


        sqv.set_global('RECT_WIDTH', random.randint(10, 20))
        sqv.set_global('RECT_HEIGHT', random.randint(10, 20))

        d = activate_agent(500, agents=reset_trained_agent, render=False)
        d = get_agent_dict(d, 0)
        d_reset.append(d)

        d = activate_agent(500, agents=nonreset_trained_agent, render=False)
        d = get_agent_dict(d, 0)
        d_nonreset.append(d)

        d = activate_agent(500, render=False)
        d = get_agent_dict(d, 0)
        d_nontrained.append(d)

    d = join_dict_list(d_reset)
    draw_plots(d, use_alpha=True, plot_locs_on_errors=False, plot_locs_on_tds=False)

    d = join_dict_list(d_nonreset)
    draw_plots(d, use_alpha=True, plot_locs_on_errors=False, plot_locs_on_tds=False)

    d = join_dict_list(d_nontrained)
    draw_plots(d, use_alpha=False, plot_locs_on_errors=False, plot_locs_on_tds=False)

    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
