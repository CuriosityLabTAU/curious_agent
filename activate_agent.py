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

    for x in range(sqv.RECT_WIDTH + 1):
        for y in range(sqv.RECT_HEIGHT + 1):
            env.agents[0]["loc"] = np.array([x, y])
            for i in range(4):
                ob, _, _, _ = env.step(np.array([0]), 0)
                values_before[i][x, y] += np.amax(agents[0].q_function.hypot(ob))

    env.agents[0]["loc"] = np.array([5, 5])

    trained_agents = []

    for timestep in xrange(number_of_epoches * epoch_time):
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
            if timestep % epoch_time == 0 and timestep != 0 and reset_agent:
                epoches_errors.append(epoch_error)
                epoch_error = []
                epoches_tds.append(epoch_td)
                epoch_td = []
                agent.reset_network()
                env.agents[i]["loc"] = env.square_space.sample()
                states[i] = env._get_all_observations()[i]
        # learner_c = agent.train(300)
        # costs.append(np.sqrt(learner_c))
        env.render()
    epoches_errors.append(epoch_error)
    epoches_tds.append(epoch_td)

    values = [np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in xrange(4)]

    for x in range(sqv.RECT_WIDTH + 1):
        for y in range(sqv.RECT_HEIGHT + 1):
            env.agents[0]["loc"] = np.array([x, y])
            for i in range(4):
                ob, _, _, _ = env.step(np.array([0]), 0)
                values[i][x, y] += np.amax(agents[0].q_function.hypot(ob))


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
