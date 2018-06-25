import square_env
from curious_agent import CuriousAgent
import gym
import numpy as np
import time
import square_env.envs as sqv
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

# Parameters
num_timesteps = 500
reset_learner = 100

# Agent hyper parameters
gamma = 0.9
learner_alpha = 0.001
q_alpha = 0.0001
epsilon = 0.1
# learner size

# For environment parameters: see SquareEnv.py file

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


def main():
    sqv.set({'name':'RECT_WIDTH','val':20})
    env = gym.make("square-v0")
    state = env.reset()[0]
    agent = CuriousAgent(0, gamma, learner_alpha, q_alpha, epsilon)
    error = 0
    tds = []
    errors = []
    timesteps = []
    rewards = []
    costs = []
    infos = []
    values_before = np.zeros((sqv.RECT_WIDTH+1, sqv.RECT_HEIGHT+1))

    for x in range(sqv.RECT_WIDTH+1):
        for y in range(sqv.RECT_HEIGHT+1):
            env.agents[0]["loc"] = np.array([x, y])
            for i in range(4):
                ob, _, _, _ = env.step(np.array([0]), 0)
                values_before[x,y] += np.amax(agent.q_function.hypot(ob))


    env.agents[0]["loc"] = np.array([5, 5])
    try:
        for timestep in xrange(num_timesteps):
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            errors.append(error)
            tds.append(td)
            rewards.append(reward)
            infos.append(info)
            timesteps.append(timestep)
            if timestep % 50 == 0:
                print "state: " + str(state)
                print "pred:" + str(np.round(prediction))
            if timestep % 1000 == 0:
                print "timestep: "+str(timestep)
            if timestep % reset_learner == 0:
                agent.reset_network()
                env.agents[0]["loc"] = env.square_space.sample()
            # learner_c = agent.train(300)
            # costs.append(np.sqrt(learner_c))
            env.render()
    except KeyboardInterrupt:
        pass

    print agent.learner_alpha
    print agent.q_alpha
    print agent.epsilon

    locs = info_to_location(infos)
    scales = loc_to_scalar(locs)
    print scales
    scales = np.array(scales)

    values = np.zeros((sqv.RECT_WIDTH+1,sqv.RECT_HEIGHT+1))

    for x in range(sqv.RECT_WIDTH+1):
        for y in range(sqv.RECT_HEIGHT+1):
            env.agents[0]["loc"] = np.array([x,y])
            for i in range(4):
                ob,_,_,_ = env.step(np.array([0]),0)
                values[x][y] += np.amax(agent.q_function.hypot(ob))

    print values
    norm_values = values - np.amin(values)
    print norm_values


    style.use('ggplot')



    plt.plot(timesteps,tds)
    plt.plot(timesteps, scales*((np.amax(tds)-1)/4))
    plt.title("TDS")
    plt.axis([min(timesteps),max(timesteps),min(tds),max(tds)])

    fig, ax = plt.subplots(1,1)
    ax.plot(timesteps, errors, label = 'error')
    ax.plot(timesteps, scales*((np.amax(errors)-1)/4), label = 'location')
    ax.set_title("Errors")
    ax.set_xlim([min(timesteps), max(timesteps)])
    ax.set_ylim([min(errors), max(errors)])
    plt.legend()


    fig, ax = plt.subplots(1,2)
    ax[0].set_title("Values")
    sns.heatmap(values, cmap=plt.cm.Blues, ax=ax[0], linewidths=.5)
    vmax, vmin = np.max(values), np.min(values_before)
    ax[1].set_title("Values Before Episode")
    sns.heatmap(values_before, vmax=vmax, vmin=vmin, cmap=plt.cm.Blues, ax=ax[1], linewidths=.5)

    # print values_before
    # norm_values_before = values_before - np.amin(values_before)
    # print norm_values_before

    # fig, ax = plt.subplots()
    # # lc is for barplot, how long the agent spent next to a wall, corner or neither.
    # _, lc = np.unique(scales, return_counts=True)
    # ax.set_title("Steps spent in each area")
    # plt.bar(np.arange(3), lc)
    # plt.xticks(np.arange(3), ('Middle', 'Wall', 'Corner'))

    plt.show()


if __name__ == "__main__":
    main()