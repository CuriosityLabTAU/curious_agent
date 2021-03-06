import square_env
import numpy as np
import gym
from curious_agent import CuriousAgent
import square_env.envs as sqv
from copy import deepcopy
import random
import stats


PRINT_STATE_PRED = 50


PRINT_TIME_STEP = 500


def activate_agent(epoch_time, number_of_epoches=1, number_of_agents=1, reset_agent=True, agents=None,
                   render=True, print_info=True, reset_env=False, env=None, get_avg_errors=False, set_cube=0,
                   get_values_field=False, moving_walls_amount=0, moving_wall_start_index=0, init_learners=None,
                   get_last_step_avg_error=False, number_of_error_agents=1):
    if env is None:
        env = gym.make('square-v0')
    states = env.reset(render=render)
    if agents is None:
        agents = []
    number_of_agents = max(number_of_agents, len(agents))
    for i in range(len(agents), number_of_agents):
        agents.append(CuriousAgent(i))

    list_of_q = []

    total_errors = [[] for _ in range(number_of_agents)]
    last_errors = [[] for _ in range(number_of_agents)]

    agent_errors = [0] * number_of_agents
    tds = [[] for _ in range(number_of_agents)]
    errors = [[] for _ in range(number_of_agents)]
    timesteps = [[] for _ in range(number_of_agents)]
    rewards = [[] for _ in range(number_of_agents)]
    costs = [[] for _ in range(number_of_agents)]
    infos = [[] for _ in range(number_of_agents)]
    epoches_errors = [[] for _ in range(number_of_agents)]
    epoch_error = [[] for _ in range(number_of_agents)]
    epoches_tds = [[] for _ in range(number_of_agents)]
    epoch_td = [[] for _ in range(number_of_agents)]
    values_before = [[np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in range(4)] for _ in range(number_of_agents)]

    for t in range(number_of_agents):
        for x in range(sqv.RECT_WIDTH + 1):
            for y in range(sqv.RECT_HEIGHT + 1):
                env.agents[t]["loc"] = np.array([x, y])
                for i in range(4):
                    ob, _, _, _ = env.step(np.array([0]), 0)
                    values_before[t][i][x, y] += np.amax(agents[t].q_function.hypot(ob))
        env.agents[t]['loc'] = np.array(sqv.INIT_LOCATIONS[t])
        env.agents[t]['dir'] = np.array(sqv.INIT_DIRECTIONS[t])

    for timestep in range(number_of_epoches * epoch_time):
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
            list_of_q.append(deepcopy(agent.q_function.layers))
            if print_info:
                if timestep % PRINT_STATE_PRED == 0:
                    print("state: " + str(state))
                    print("prediction: " + str(np.round(prediction)))
                if timestep % PRINT_TIME_STEP == 0:
                    print("time step: " + str(timestep))
            if timestep % epoch_time == 0 and timestep != 0:
                epoches_errors[i].append(epoch_error[i])
                epoch_error[i] = []
                epoches_tds[i].append(epoch_td[i])
                epoch_td[i] = []
                if get_last_step_avg_error and i < number_of_error_agents:
                    last_errors[i].append(stats.average_errors_on_trained_agent(agent, env))
                if reset_agent:
                    agent.reset_network()
                    if init_learners is not None:
                        agent.learner = deepcopy(init_learners[i])
                    env.agents[i]["loc"] = env.square_space.sample()
                    states[i] = env._get_all_observations()[i]
            if reset_env and i + 1 == len(agents) and timestep % epoch_time == 0 and timestep != 0:
                if render:
                    env.close()
                sqv.set_global('RECT_WIDTH', random.randint(15, 15))
                sqv.set_global('RECT_HEIGHT', random.randint(15, 15))
                env = gym.make('square-v0')
                states = env.reset(render=render)
                for c in range(moving_walls_amount):
                    sqv.INIT_LOCATIONS[c + moving_wall_start_index] = env.square_space.sample()
                    sqv.INIT_DIRECTIONS[c + moving_wall_start_index] = random.choice(stats.ALL_DIRECTIONS)
                for c in range(set_cube):
                    sqv.INIT_LOCATIONS[c + number_of_agents + moving_walls_amount] = env.square_space.sample()
            if get_avg_errors and i < number_of_error_agents:
                total_errors[i].append(stats.func2(agent, env))
        # learner_c = agent.train(300)
        # costs.append(np.sqrt(learner_c))
        if render:
            env.render()
    for i in range(number_of_agents):
        epoches_errors[i].append(epoch_error[i])
        epoches_tds[i].append(epoch_td[i])

    values = [[np.zeros((sqv.RECT_WIDTH + 1, sqv.RECT_HEIGHT + 1)) for _ in range(4)] for _ in range(number_of_agents)]

    for t in range(number_of_agents):
        for x in range(sqv.RECT_WIDTH + 1):
            for y in range(sqv.RECT_HEIGHT + 1):
                env.agents[t]["loc"] = np.array([x, y])
                for i in range(4):
                    ob, _, _, _ = env.step(np.array([0]), 0)
                    values[t][i][x, y] += np.amax(agents[t].q_function.hypot(ob))

    if render:
        env.close()

    ret = {}
    if get_values_field:
        q = []
        cs = []
        for i in agents:
            v, c = stats.get_agent_value_field(i, env)
            q.append(v)
            cs.append(c)
        ret['fields'] = q
        ret['fields_colors'] = cs

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
    ret['total_errors'] = total_errors
    ret['last_errors'] = last_errors

    return ret
