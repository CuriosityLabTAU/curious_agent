import square_env
from curious_agent import CuriousAgent
import gym
import numpy as np
import time
import square_env.envs as sqv
import matplotlib.pyplot as plt
from matplotlib import style
import threading

def loc_to_scalar(loc):
    scales = []
    for i in loc:
        if (i[0] == sqv.RECT_WIDTH and (i[1] == sqv.RECT_HEIGHT or i[1] == 0)) or (i[0] == 0 and (i[1] == sqv.RECT_HEIGHT or i[1] == 0)):
            scales.append(3)
        elif i[0] == sqv.RECT_WIDTH or i[1] == sqv.RECT_HEIGHT or i[0] == 0 or i[1] == 0:
            scales.append(2)
        else:
            scales.append(1)
    return scales

def info_to_location(info):
    loc = []

    for i in info:
        loc.append(i[0]["loc"])
    return loc


def draw_plots(values_dict, plot_tds=True, plot_errors=True, plot_locations_bars=True, plot_values=True,
               plot_values_before=True, plot_locs_on_tds=True, plot_locs_on_errors=True, use_alpha=True):
    tds = values_dict['tds']
    errors = values_dict['errors']
    timesteps = values_dict['timesteps']
    rewards = values_dict['rewards']
    costs = values_dict['costs']
    infos = values_dict['infos']
    epoches_errors = values_dict['epoches_errors']
    epoches_tds = values_dict['epoches_tds']
    values_before = values_dict['values_before']
    values = values_dict['values']

    locs = info_to_location(infos)
    scales = loc_to_scalar(locs)
    scales = np.array(scales)

    epoches_factor = max(255/len(epoches_tds), 1)

    style.use('ggplot')

    lc = np.zeros(3)
    for i in scales:
        lc[i - 1] += 1

    colormap = plt.cm.jet

    if plot_tds:
        plt.plot(timesteps, tds)
        if plot_locs_on_tds:
            plt.plot(timesteps, scales*((np.amax(tds)-1)/4))
        plt.title("TDS")
        plt.axis([min(timesteps), max(timesteps), min(tds), max(tds)])

        fig, ax = plt.subplots(1, 1)

        for i, td in enumerate(epoches_tds):
            ax.plot(np.arange(len(td)), td, label='epoch: ' + str(i), c=colormap(i*epoches_factor),
                    alpha=min(1.0 / (len(epoches_errors) - i) + 0.1, 1.0) if use_alpha else 1.0)
        ax.set_title("TDs")
        plt.legend()

    if plot_errors:
        fig, ax = plt.subplots(1, 1)
        ax.plot(timesteps, errors, label='error')
        if plot_locs_on_errors:
            ax.plot(timesteps, scales*((np.amax(errors)-1)/4), label = 'location')
        ax.set_title("Errors")
        ax.set_xlim([min(timesteps), max(timesteps)])
        ax.set_ylim([min(errors), max(errors)])
        plt.legend()

        fig, ax = plt.subplots(1, 1)
        for i, error in enumerate(epoches_errors):
            ax.plot(np.arange(len(error)), error, label='epoch: '+str(i), c=colormap(i*epoches_factor),
                    alpha=min(1.0/(len(epoches_errors)-i)+0.1, 1.0) if use_alpha else 1.0)
        ax.set_title("Errors")
        plt.legend()



    style.use("classic")

    if plot_values:
        for ind, k in enumerate(values):
            fig, ax = plt.subplots()
            ax.set_title("Values ("+str(ind)+")")
            ax.matshow(k,cmap=plt.cm.Blues)

            for i in xrange(sqv.RECT_WIDTH+1):
                for j in xrange(sqv.RECT_HEIGHT+1):
                    c = k[i, j]
                    ax.text(j, i, str(c)[:5], va='center', ha='center',size=8)

    if plot_values_before:
        for ind, k in enumerate(values_before):
            fig, ax = plt.subplots()
            ax.matshow(k, cmap=plt.cm.Reds)
            ax.set_title("Values Before Episode ("+str(ind)+")")

            for i in xrange(sqv.RECT_WIDTH+1):
                for j in xrange(sqv.RECT_HEIGHT+1):
                    c = k[i, j]
                    ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    style.use("bmh")

    if plot_locations_bars:
        fig, ax = plt.subplots()

        plt.bar(np.arange(3), lc)
        plt.xticks(np.arange(3), ('Middle', 'Wall', 'Corner'))

    plt.show()