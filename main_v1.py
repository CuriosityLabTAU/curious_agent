import square_env
from curious_agent import CuriousAgent
import gym
import numpy as np
import time
import square_env.envs as sqv
import matplotlib.pyplot as plt
from matplotlib import style

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
    env = gym.make("square-v0")
    state = env.reset()[0]
    agent = CuriousAgent(0)
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
        for timestep in xrange(50000):
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
            if timestep % 100 == 0:
                agent.reset_network()
                env.agents[0]["loc"] = env.square_space.sample()
            #learner_c = agent.train(300)
            #costs.append(np.sqrt(learner_c))
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

    lc = np.zeros(3)
    for i in scales:
        lc[i - 1] += 1

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

    style.use("classic")

    fig, ax = plt.subplots()
    ax.set_title("Values")
    ax.matshow(values,cmap=plt.cm.Blues)

    for i in xrange(sqv.RECT_WIDTH+1):
        for j in xrange(sqv.RECT_HEIGHT+1):
            c = norm_values[i][j]
            ax.text(j, i, str(c)[:5], va='center', ha='center',size=8)

    print values_before
    norm_values_before = values_before - np.amin(values_before)
    print norm_values_before
    fig, ax = plt.subplots()
    ax.matshow(values_before, cmap=plt.cm.Reds)
    ax.set_title("Values Before Episode")

    for i in xrange(sqv.RECT_WIDTH+1):
        for j in xrange(sqv.RECT_HEIGHT+1):
            c = norm_values_before[i, j]
            ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    style.use("bmh")

    fig, ax = plt.subplots()

    plt.bar(np.arange(3), lc)
    plt.xticks(np.arange(3), ('Middle', 'Wall', 'Corner'))

    plt.show()






if __name__ == "__main__":
    main()