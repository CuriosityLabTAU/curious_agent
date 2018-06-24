import square_env
from curious_agent import CuriousAgent
import gym
import numpy as np
import time

def loc_to_scalar(loc):
    scales = []
    for i in loc:
        if (i[0] == 10 and (i[1] == 10 or i[1] == 0)) or (i[0] == 0 and (i[1] == 10 or i[1] == 0)):
            scales.append(3)
        elif i[0] == 10 or i[1] == 10 or i[0] == 0 or i[1] == 0:
            scales.append(2)
        else:
            scales.append(1)
    return scales

def info_to_location(info):
    loc = []
    for i in info:
        loc.append(i["location"])
    return loc


def main():
    env = gym.make("square-v0")
    state = env.reset()
    agent = CuriousAgent()
    error = 1.0
    tds = []
    errors = []
    timesteps = []
    rewards = []
    costs = []
    infos = []

    for timestep in xrange(200):
        state, error, info, td, reward, prediction = agent.take_step(env, state, error)
        errors.append(error)
        tds.append(td)
        rewards.append(reward)
        infos.append(info)
        timesteps.append(timestep)
        if timestep % 50 == 0:
            print "state: " + str(state)
            print "pred:" + str(np.round(prediction))
        #learner_c = agent.train(300)
        #costs.append(np.sqrt(learner_c))
        env.render()

    print agent.learner_alpha
    print agent.value_alpha
    print agent.epsilon

    locs = info_to_location(infos)
    scales = loc_to_scalar(locs)
    print scales
    import matplotlib.pyplot as plt
    plt.figure(timesteps, scales)
    plt.plot(timesteps,tds)
    plt.title("TDS")
    plt.axis([min(timesteps),max(timesteps),min(tds),max(tds)])

    fig, ax = plt.subplots(1,1)
    ax.plot(timesteps, errors)


    ax.plot(timesteps)
    ax.set_title("Errors")
    ax([min(timesteps), max(timesteps), min(errors), max(errors)])

    plt.figure()
    plt.plot(timesteps, rewards)
    plt.title("Rewards")
    plt.axis([min(timesteps), max(timesteps), min(rewards), max(rewards)])

    plt.show()







if __name__ == "__main__":
    main()