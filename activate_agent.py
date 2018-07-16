import square_env
import numpy as np
import gym
from curious_agent import CuriousAgent
import square_env.envs as sqv


PRINT_STATE_PRED = 50


PRINT_TIME_STEP = 500


def activate_agent(epoch_time, number_of_epoches=1, number_of_agents=1, reset_agent=True):
    env = gym.make('square-v0')
    states = env.reset()
    agents = []
    for i in xrange(number_of_agents):
        agents.append(CuriousAgent(i))

    agent_errors = [0] * number_of_agents
    tds = [[] for _ in xrange(number_of_agents)]
    errors = [[] for _ in xrange(number_of_agents)]
    timesteps = [[] for _ in xrange(number_of_agents)]
    rewards = [[] for _ in xrange(number_of_agents)]
    costs = [[] for _ in xrange(number_of_agents)]
    infos = [[] for _ in xrange(number_of_agents)]
    epoches_errors = [[] for _ in xrange(number_of_agents)]
    epoch_error = [[] for _ in xrange(number_of_agents)]
    epoches_tds = [[] for _ in xrange(number_of_agents)]
    epoch_td = [[] for _ in xrange(number_of_agents)]
    values_before = [[np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in xrange(4)] for _ in xrange(number_of_agents)]

    for t in xrange(number_of_agents):
        for x in range(sqv.RECT_WIDTH + 1):
            for y in range(sqv.RECT_HEIGHT + 1):
                env.agents[t]["loc"] = np.array([x, y])
                for i in range(4):
                    ob, _, _, _ = env.step(np.array([0]), 0)
                    values_before[t][i][x, y] += np.amax(agents[t].q_function.hypot(ob))
        env.agents[t]['loc'] = sqv.INIT_LOCATIONS[t]
        env.agents[t]['dir'] = sqv.INIT_DIRECTIONS[t]

    for timestep in xrange(number_of_epoches * epoch_time):
        for i, agent in enumerate(agents):
            state = states[i]
            error = agent_errors[i]
            state, error, info, td, reward, prediction = agent.take_step(env, state, error)
            states[i] = state
            agent_errors[i] = error
            errors[i].append(error)
            tds[i].append(td)
            rewards[i].append(reward)
            infos[i].append(info)
            timesteps[i].append(timestep)
            epoch_td[i].append(td)
            epoch_error[i].append(error)
            if timestep % PRINT_STATE_PRED == 0:
                print "state: " + str(state)
                print "prediction: " + str(np.round(prediction))
            if timestep % PRINT_TIME_STEP == 0:
                print "time step: " + str(timestep)
            if timestep % epoch_time == 0 and timestep != 0 and reset_agent:
                epoches_errors[i].append(epoch_error)
                epoch_error[i] = []
                epoches_tds[i].append(epoch_td)
                epoch_td[i] = []
                agent.reset_network()
                env.agents[i]["loc"] = env.square_space.sample()
                states[i] = env._get_all_observations()[i]
        # learner_c = agent.train(300)
        # costs.append(np.sqrt(learner_c))
        env.render()
    for i in xrange(number_of_agents):
        epoches_errors[i].append(epoch_error[i])
        epoches_tds[i].append(epoch_td[i])

    values = [[np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in xrange(4)] for _ in xrange(number_of_agents)]

    for t in xrange(number_of_agents):
        for x in range(sqv.RECT_WIDTH + 1):
            for y in range(sqv.RECT_HEIGHT + 1):
                env.agents[t]["loc"] = np.array([x, y])
                for i in range(4):
                    ob, _, _, _ = env.step(np.array([0]), 0)
                    values[t][i][x, y] += np.amax(agents[t].q_function.hypot(ob))


    ret = {}
    ret['agents'] = agents
    ret['tds'] = tds
    ret['errors'] = errors
    ret['timesteps'] = timesteps
    ret['rewards'] = rewards
    ret['costs'] = costs
    ret['infos'] = infos
    ret['epoches_errors'] = epoches_errors
    ret['epoches_tds'] = epoches_tds
    ret['values_before'] = values_before
    ret['values'] = values

    return ret
