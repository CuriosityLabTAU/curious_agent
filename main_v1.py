import square_env
from curious_agent import CuriousAgent
import gym
import numpy as np
import time
import square_env.envs as sqv
import matplotlib.pyplot as plt
from matplotlib import style
import threading
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


def main():

    d = activate_agent(100,20)
    d = get_agent_dict(d, 0)
    draw_plots(d)



    from IPython import embed
    embed()


if __name__ == "__main__":
    main()