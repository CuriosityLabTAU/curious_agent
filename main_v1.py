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
from draw_plots import draw_plots, plot_together
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

    d_nonreset = {}
    d_reset = {}
    d_nontrained = {}

    for i in xrange(500):
        d = activate_agent(50, 25, render=False, print_info=False)
        reset_trained_agent = d['agents']
        reset_trained_agent[0].reset_network()


        d = activate_agent(50*25, reset_agent=False, render=False, print_info=False)
        nonreset_trained_agent = d['agents']
        nonreset_trained_agent[0].reset_network()


        sqv.set_global('RECT_WIDTH', random.randint(10, 20))
        sqv.set_global('RECT_HEIGHT', random.randint(10, 20))

        d = activate_agent(100, agents=reset_trained_agent, render=False, print_info=False)
        d = get_agent_dict(d, 0)
        if i == 0:
            for j in d:
                d_reset[j] = np.array(d[j]) if isinstance(d[j], list) else d[j]
        else:
            for j in d_reset:
                if isinstance(d_reset[j], np.ndarray) and d_reset[j].dtype == 'float':
                    d_reset[j] = (float(i) * d_reset[j] + np.array(d[j]))/float(i+1)

        d = activate_agent(100, agents=nonreset_trained_agent, render=False, print_info=False)
        d = get_agent_dict(d, 0)
        if i == 0:
            for j in d:
                d_nonreset[j] = np.array(d[j]) if isinstance(d[j], list) else d[j]
        else:
            for j in d_nonreset:
                if isinstance(d_nonreset[j], np.ndarray) and d_nonreset[j].dtype == 'float':
                    d_nonreset[j] = (float(i)*d_nonreset[j] + np.array(d[j]))/float(i+1)

        d = activate_agent(100, render=False, print_info=False)
        d = get_agent_dict(d, 0)
        if i == 0:
            for j in d:
                d_nontrained[j] = np.array(d[j]) if isinstance(d[j], list) else d[j]
        else:
            for j in d_nontrained:
                if isinstance(d_nontrained[j], np.ndarray) and d_nontrained[j].dtype == 'float':
                    d_nontrained[j] = (float(i)*d_nontrained[j] + np.array(d[j]))/float(i+1)

        print "finished running #%i"%i

    #draw_plots(d_reset, use_alpha=True, plot_locs_on_errors=False, plot_locs_on_tds=False, plot_values=False, plot_values_before=False)

    #draw_plots(d_nonreset, use_alpha=True, plot_locs_on_errors=False, plot_locs_on_tds=False, plot_values=False, plot_values_before=False)

    #draw_plots(d_nontrained, use_alpha=False, plot_locs_on_errors=False, plot_locs_on_tds=False, plot_values=False, plot_values_before=False)

    plot_together(d_nontrained['timesteps'], [d_nontrained['errors'], 'non trained'],
                  [d_nonreset['errors'], 'non reset'], [d_reset['errors'], 'reset'], title='Errors')

    plot_together(d_nontrained['timesteps'], [d_nontrained['tds'], 'non trained'],
                  [d_nonreset['tds'], 'non reset'], [d_reset['tds'], 'reset'], title='Errors')

    from IPython import embed
    embed()


if __name__ == "__main__":
    main()
