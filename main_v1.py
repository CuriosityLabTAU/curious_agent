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


NUM_OF_EPOCHES = 50


PRINT_STATE_PRED = 50


PRINT_TIME_STEP = 500


EPOCH_TIME = 20


NUMBER_OF_AGENTS = 1


def main():

    env = gym.make('square-v0')
    states = env.reset()
    agents = []
    for i in xrange(NUMBER_OF_AGENTS):
        agents.append(CuriousAgent(i))

    agent_errors = [0]*NUMBER_OF_AGENTS
    tds = []
    errors = []
    timesteps = []
    rewards = []
    costs = []
    infos = []
    epoches_errors = []
    epoch_error = []
    epoches_tds = []
    epoch_td = []
    values_before = [np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in xrange(4)]

    for x in range(sqv.RECT_WIDTH+1):
        for y in range(sqv.RECT_HEIGHT+1):
            env.agents[0]["loc"] = np.array([x, y])
            for i in range(4):
                ob, _, _, _ = env.step(np.array([0]), 0)
                values_before[i][x, y] += np.amax(agents[0].q_function.hypot(ob))

    env.agents[0]["loc"] = np.array([5, 5])

    trained_agents = []

    for timestep in xrange(NUM_OF_EPOCHES*EPOCH_TIME):
        for i, agent in enumerate(agents):
            state = states[i]
            error = agent_errors[i]
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            states[i] = state
            agent_errors[i] = error
            errors.append(error)
            tds.append(td)
            rewards.append(reward)
            infos.append(info)
            timesteps.append(timestep)
            epoch_td.append(td)
            epoch_error.append(error)
            if timestep % PRINT_STATE_PRED == 0:
                print "state: " + str(state)
                print "prediction: " + str(np.round(prediction))
            if timestep % PRINT_TIME_STEP == 0:
                print "time step: "+str(timestep)
            if timestep % EPOCH_TIME == 0 and timestep != 0:
                epoches_errors.append(epoch_error)
                epoch_error = []
                epoches_tds.append(epoch_td)
                epoch_td = []
                agent.reset_network()
                env.agents[i]["loc"] = env.square_space.sample()
                states[i] = env._get_all_observations()[i]
        #learner_c = agent.train(300)
        #costs.append(np.sqrt(learner_c))
        env.render()
    epoches_errors.append(epoch_error)
    epoches_tds.append(epoch_td)


    trained_agents.append(agents[0])
    agents[0] = CuriousAgent(0)

    for timestep in xrange(NUM_OF_EPOCHES*EPOCH_TIME):
        for i, agent in enumerate(agents):
            state = states[i]
            error = agent_errors[i]
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            states[i] = state
            agent_errors[i] = error
            errors.append(error)
            tds.append(td)
            rewards.append(reward)
            infos.append(info)
            timesteps.append(timestep)
            epoch_td.append(td)
            epoch_error.append(error)
            if timestep % PRINT_STATE_PRED == 0:
                print "state: " + str(state)
                print "prediction: " + str(np.round(prediction))
            if timestep % PRINT_TIME_STEP == 0:
                print "time step: "+str(timestep)
        #learner_c = agent.train(300)
        #costs.append(np.sqrt(learner_c))
        env.render()
    epoches_errors.append(epoch_error)
    epoches_tds.append(epoch_td)

    epoches_errors = []
    epoches_tds = []

    agent_errors = [0] * NUMBER_OF_AGENTS
    tds = []
    errors = []
    timesteps = []
    rewards = []
    costs = []
    infos = []
    epoches_errors = []
    epoch_error = []
    epoches_tds = []
    epoch_td = []

    sqv.set_global("RECT_WIDTH", random.randint(10,20))
    sqv.set_global("RECT_HEIGHT", random.randint(10, 20))

    env = gym.make('square-v0')
    states = env.reset()

    agents[0].reset_network()


    for timestep in xrange(500):
        for i, agent in enumerate(agents):
            state = states[i]
            error = agent_errors[i]
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            states[i] = state
            agent_errors[i] = error
            errors.append(error)
            tds.append(td)
            rewards.append(reward)
            infos.append(info)
            timesteps.append(timestep)
            epoch_td.append(td)
            epoch_error.append(error)
            if timestep % PRINT_STATE_PRED == 0:
                print "state: " + str(state)
                print "prediction: " + str(np.round(prediction))
            if timestep % PRINT_TIME_STEP == 0:
                print "time step: "+str(timestep)
        #learner_c = agent.train(300)
        #costs.append(np.sqrt(learner_c))

    print agents[0].learner_alpha
    print agents[0].q_alpha
    print agents[0].epsilon

    locs = info_to_location(infos)
    scales = loc_to_scalar(locs)
    print scales
    scales = np.array(scales)
    first_errors = errors
    values = [np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in xrange(4)]

    for x in range(sqv.RECT_WIDTH + 1):
        for y in range(sqv.RECT_HEIGHT + 1):
            env.agents[0]["loc"] = np.array([x, y])
            for i in range(4):
                ob, _, _, _ = env.step(np.array([0]), 0)
                values[i][x, y] += np.amax(agents[0].q_function.hypot(ob))

    print values
    norm_values = values - np.amin(values)
    print norm_values

    style.use('ggplot')

    lc = np.zeros(3)
    for i in scales:
        lc[i - 1] += 1

    plt.plot(timesteps, tds)
    plt.plot(timesteps, scales * ((np.amax(tds) - 1) / 4))
    plt.title("TDS")
    plt.axis([min(timesteps), max(timesteps), min(tds), max(tds)])

    fig, ax = plt.subplots(1, 1)
    ax.plot(timesteps, errors, label='error')
    ax.plot(timesteps, scales * ((np.amax(errors) - 1) / 4), label='location')
    ax.set_title("Errors")
    ax.set_xlim([min(timesteps), max(timesteps)])
    ax.set_ylim([min(errors), max(errors)])
    plt.legend()

    colormap = plt.cm.jet

    fig, ax = plt.subplots(1, 1)
    for i, error in enumerate(epoches_errors):
        ax.plot(np.arange(len(error)), error, label='epoch: ' + str(i), c=colormap(i),
                alpha=min(1.0 / (len(epoches_errors) - i) + 0.1, 1.0))
    ax.set_title("Errors")
    plt.legend()

    fig, ax = plt.subplots(1, 1)

    for i, td in enumerate(epoches_tds):
        ax.plot(np.arange(len(td)), td, label='epoch: ' + str(i), c=colormap(i),
                alpha=min(1.0 / (len(epoches_errors) - i) + 0.1, 1.0))
    ax.set_title("TDs")
    plt.legend()

    style.use("classic")

    for ind, k in enumerate(values):
        fig, ax = plt.subplots()
        ax.set_title("Values (" + str(ind) + ")")
        ax.matshow(k, cmap=plt.cm.Blues)

        for i in xrange(sqv.RECT_WIDTH + 1):
            for j in xrange(sqv.RECT_HEIGHT + 1):
                c = k[i, j]
                ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    for ind, k in enumerate(values_before):
        print k
        fig, ax = plt.subplots()
        ax.matshow(k, cmap=plt.cm.Reds)
        ax.set_title("Values Before Episode (" + str(ind) + ")")

        for i in xrange(sqv.RECT_WIDTH + 1):
            for j in xrange(sqv.RECT_HEIGHT + 1):
                c = k[i, j]
                ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    style.use("bmh")

    fig, ax = plt.subplots()

    plt.bar(np.arange(3), lc)
    plt.xticks(np.arange(3), ('Middle', 'Wall', 'Corner'))

    plt.show()

    epoches_errors = []
    epoches_tds = []

    agent_errors = [0] * NUMBER_OF_AGENTS
    tds = []
    errors = []
    timesteps = []
    rewards = []
    costs = []
    infos = []
    epoches_errors = []
    epoch_error = []
    epoches_tds = []
    epoch_td = []

    agents[0] = trained_agents[0]
    agents[0].reset_network()

    states = env.reset()

    for timestep in xrange(500):
        for i, agent in enumerate(agents):
            state = states[i]
            error = agent_errors[i]
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            states[i] = state
            agent_errors[i] = error
            errors.append(error)
            tds.append(td)
            rewards.append(reward)
            infos.append(info)
            timesteps.append(timestep)
            epoch_td.append(td)
            epoch_error.append(error)
            if timestep % PRINT_STATE_PRED == 0:
                print "state: " + str(state)
                print "prediction: " + str(np.round(prediction))
            if timestep % PRINT_TIME_STEP == 0:
                print "time step: " + str(timestep)
        # learner_c = agent.train(300)
        # costs.append(np.sqrt(learner_c))

    second_errors = errors

    print agents[0].learner_alpha
    print agents[0].q_alpha
    print agents[0].epsilon

    locs = info_to_location(infos)
    scales = loc_to_scalar(locs)
    print scales
    scales = np.array(scales)

    values = [np.zeros((sqv.RECT_WIDTH+1, sqv.RECT_HEIGHT+1)) for _ in xrange(4)]

    for x in range(sqv.RECT_WIDTH+1):
        for y in range(sqv.RECT_HEIGHT+1):
            env.agents[0]["loc"] = np.array([x,y])
            for i in range(4):
                ob,_,_,_ = env.step(np.array([0]),0)
                values[i][x, y] += np.amax(agents[0].q_function.hypot(ob))

    print values
    norm_values = values - np.amin(values)
    print norm_values

    style.use('ggplot')

    lc = np.zeros(3)
    for i in scales:
        lc[i - 1] += 1

    plt.plot(timesteps, tds)
    plt.plot(timesteps, scales*((np.amax(tds)-1)/4))
    plt.title("TDS")
    plt.axis([min(timesteps), max(timesteps), min(tds), max(tds)])

    fig, ax = plt.subplots(1, 1)
    ax.plot(timesteps, errors, label='error')
    ax.plot(timesteps, first_errors, label='first errors')
    ax.plot(timesteps, scales*((np.amax(errors)-1)/4), label = 'location')
    ax.set_title("Errors")
    ax.set_xlim([min(timesteps), max(timesteps)])
    ax.set_ylim([min(errors), max(errors)])
    plt.legend()


    colormap = plt.cm.jet

    fig, ax = plt.subplots(1, 1)
    for i, error in enumerate(epoches_errors):
        ax.plot(np.arange(len(error)), error, label='epoch: '+str(i), c=colormap(i), alpha=min(1.0/(len(epoches_errors)-i)+0.1, 1.0))
    ax.set_title("Errors")
    plt.legend()


    fig, ax = plt.subplots(1, 1)

    for i, td in enumerate(epoches_tds):
        ax.plot(np.arange(len(td)), td, label='epoch: ' + str(i), c=colormap(i), alpha=min(1.0/(len(epoches_errors)-i)+0.1, 1.0))
    ax.set_title("TDs")
    plt.legend()


    style.use("classic")

    for ind, k in enumerate(values):
        fig, ax = plt.subplots()
        ax.set_title("Values ("+str(ind)+")")
        ax.matshow(k,cmap=plt.cm.Blues)

        for i in xrange(sqv.RECT_WIDTH+1):
            for j in xrange(sqv.RECT_HEIGHT+1):
                c = k[i, j]
                ax.text(j, i, str(c)[:5], va='center', ha='center',size=8)

    for ind, k in enumerate(values_before):
        print k
        fig, ax = plt.subplots()
        ax.matshow(k, cmap=plt.cm.Reds)
        ax.set_title("Values Before Episode ("+str(ind)+")")

        for i in xrange(sqv.RECT_WIDTH+1):
            for j in xrange(sqv.RECT_HEIGHT+1):
                c = k[i, j]
                ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    style.use("bmh")

    fig, ax = plt.subplots()

    plt.bar(np.arange(3), lc)
    plt.xticks(np.arange(3), ('Middle', 'Wall', 'Corner'))

    plt.show()

    agent_errors = [0] * NUMBER_OF_AGENTS
    tds = []
    errors = []
    timesteps = []
    rewards = []
    costs = []
    infos = []
    epoches_errors = []
    epoch_error = []
    epoches_tds = []
    epoch_td = []

    agents[0] = CuriousAgent(0)

    states = env.reset()

    for timestep in xrange(500):
        for i, agent in enumerate(agents):
            state = states[i]
            error = agent_errors[i]
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            states[i] = state
            agent_errors[i] = error
            errors.append(error)
            tds.append(td)
            rewards.append(reward)
            infos.append(info)
            timesteps.append(timestep)
            epoch_td.append(td)
            epoch_error.append(error)
            if timestep % PRINT_STATE_PRED == 0:
                print "state: " + str(state)
                print "prediction: " + str(np.round(prediction))
            if timestep % PRINT_TIME_STEP == 0:
                print "time step: " + str(timestep)
        # learner_c = agent.train(300)
        # costs.append(np.sqrt(learner_c))


    print agents[0].learner_alpha
    print agents[0].q_alpha
    print agents[0].epsilon

    locs = info_to_location(infos)
    scales = loc_to_scalar(locs)
    print scales
    scales = np.array(scales)

    values = [np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in xrange(4)]

    for x in range(sqv.RECT_WIDTH + 1):
        for y in range(sqv.RECT_HEIGHT + 1):
            env.agents[0]["loc"] = np.array([x, y])
            for i in range(4):
                ob, _, _, _ = env.step(np.array([0]), 0)
                values[i][x, y] += np.amax(agents[0].q_function.hypot(ob))

    print values
    norm_values = values - np.amin(values)
    print norm_values

    style.use('ggplot')

    lc = np.zeros(3)
    for i in scales:
        lc[i - 1] += 1

    plt.plot(timesteps, tds)
    plt.plot(timesteps, scales * ((np.amax(tds) - 1) / 4))
    plt.title("TDS")
    plt.axis([min(timesteps), max(timesteps), min(tds), max(tds)])

    fig, ax = plt.subplots(1, 1)
    ax.plot(timesteps, errors, label='error')
    ax.plot(timesteps, first_errors, label='first errors')
    ax.plot(timesteps, second_errors, label='second errors')
    ax.set_title("Errors")
    ax.set_xlim([min(timesteps), max(timesteps)])
    ax.set_ylim([min(errors), max(errors)])
    plt.legend()

    colormap = plt.cm.jet

    fig, ax = plt.subplots(1, 1)
    for i, error in enumerate(epoches_errors):
        ax.plot(np.arange(len(error)), error, label='epoch: ' + str(i), c=colormap(i),
                alpha=min(1.0 / (len(epoches_errors) - i) + 0.1, 1.0))
    ax.set_title("Errors")
    plt.legend()

    fig, ax = plt.subplots(1, 1)

    for i, td in enumerate(epoches_tds):
        ax.plot(np.arange(len(td)), td, label='epoch: ' + str(i), c=colormap(i),
                alpha=min(1.0 / (len(epoches_errors) - i) + 0.1, 1.0))
    ax.set_title("TDs")
    plt.legend()

    style.use("classic")

    for ind, k in enumerate(values):
        fig, ax = plt.subplots()
        ax.set_title("Values (" + str(ind) + ")")
        ax.matshow(k, cmap=plt.cm.Blues)

        for i in xrange(sqv.RECT_WIDTH + 1):
            for j in xrange(sqv.RECT_HEIGHT + 1):
                c = k[i, j]
                ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    for ind, k in enumerate(values_before):
        print k
        fig, ax = plt.subplots()
        ax.matshow(k, cmap=plt.cm.Reds)
        ax.set_title("Values Before Episode (" + str(ind) + ")")

        for i in xrange(sqv.RECT_WIDTH + 1):
            for j in xrange(sqv.RECT_HEIGHT + 1):
                c = k[i, j]
                ax.text(j, i, str(c)[:5], va='center', ha='center', size=8)

    style.use("bmh")

    fig, ax = plt.subplots()

    plt.bar(np.arange(3), lc)
    plt.xticks(np.arange(3), ('Middle', 'Wall', 'Corner'))

    plt.show()

    from IPython import embed
    embed()





if __name__ == "__main__":
    main()