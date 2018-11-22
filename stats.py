import numpy as np
import square_env
import gym
import curious_agent as cru
from curious_agent import ALL_ACTIONS
import square_env.envs as sqv
from copy import deepcopy
from time import time

ALL_DIRECTIONS = np.array([[0, 1],[1, 0],[-1, 0], [0, -1]])

TRAINS, LABELS =  None, None

# def train_and_average_errors(agent, epoch_time=1000, render=False, print_info=False):
#     env = gym.make('square-v0')
#     agent.reset_network()
#     activate_agent(epoch_time, reset_agent=False, reset_env=False, env=env, render=render, print_info=print_info)
#     cost_avg = 0.0
#     for x in range(sqv.RECT_WIDTH):
#         for y in range(sqv.RECT_HEIGHT):
#             for direction in ALL_DIRECTIONS:
#                 for action in ALL_ACTIONS:
#                     old_state = env._get_all_observations()[agent.index]
#                     env.agents[agent.index]['loc'] = np.array([x, y])
#                     env.agents[agent.index]['dir'] = np.copy(direction)
#                     state, _, _, _ = env.step(action=action, index=agent.index)
#                     cost_avg += agent.learner.cost(np.array([np.concatenate((old_state, action))]),
#                                                    np.array([state])) / float(sqv.RECT_WIDTH*sqv.RECT_HEIGHT*12)
#     return cost_avg

def func2(agent, env):
    if TRAINS is None:
        __init__()
    return agent.learner.cost(TRAINS, LABELS)

def average_errors_on_trained_agent(agent, env):
    env = deepcopy(env)
    cost_avg = 0.0
    for x in range(sqv.RECT_WIDTH):
        for y in range(sqv.RECT_HEIGHT):
            for direction in ALL_DIRECTIONS:
                for action in ALL_ACTIONS:
                    old_state = env._get_all_observations()[agent.index]
                    env.agents[agent.index]['loc'] = np.array([x, y])
                    env.agents[agent.index]['dir'] = np.copy(direction)
                    state, _, _, _ = env.step(action=np.array([np.amax(action)]), index=agent.index)
                    cost_avg += agent.learner.cost(np.array([np.concatenate((old_state, action))]),
                                                   np.array([state])) / float(sqv.RECT_WIDTH*sqv.RECT_HEIGHT*12)
    return cost_avg


def load_train_and_labels(agent, env):
    global TRAINS, LABELS
    env = deepcopy(env)
    cost_avg = 0.0
    trains = []
    labels = []
    for x in range(sqv.RECT_WIDTH):
        for y in range(sqv.RECT_HEIGHT):
            for direction in ALL_DIRECTIONS:
                for action in ALL_ACTIONS:
                    old_state = env._get_all_observations()[agent.index]
                    env.agents[agent.index]['loc'] = np.array([x, y])
                    env.agents[agent.index]['dir'] = np.copy(direction)
                    state, _, _, _ = env.step(action=np.array([np.amax(action)]), index=agent.index)
                    trains.append(np.array([np.concatenate((old_state, action))]).flatten())
                    labels.append(np.array([state]).flatten())
    TRAINS = np.array(trains, dtype=float)
    LABELS = np.array(labels, dtype=float)
    return cost_avg


def __init__():
    agent_count = sqv.AGENTS_COUNT
    sqv.set_global("AGENTS_COUNT", 1)
    import curious_agent
    env = gym.make('square-v0')
    env.reset(render=False)
    load_train_and_labels(curious_agent.CuriousAgent(0), env)
    sqv.set_global("AGENTS_COUNT", agent_count)

def get_agent_value_field(agent, env):
    env = deepcopy(env)
    x, y = np.meshgrid(np.arange(sqv.RECT_WIDTH), np.arange(sqv.RECT_HEIGHT))
    u, v = np.zeros_like(x), np.zeros_like(x)
    c = np.zeros((x.size, 4))
    for i in range(sqv.RECT_WIDTH):
        for j in range(sqv.RECT_HEIGHT):
            if env._collides(np.array([i, j])):
                c[i*sqv.RECT_WIDTH + j] = np.array([0, 0.5, 0, 1])
            else:
                c[i*sqv.RECT_WIDTH + j] = np.array([0, 0, 0, 1])
                env.agents[agent.index]['loc'] = np.array([i, j])
                m = 0.0
                for d in ALL_DIRECTIONS:
                    env.agents[agent.index]['dir'] = np.array(d)
                    state = env._get_all_observations()[agent.index]
                    val = agent.q_function.hypot(np.array([state]))
                    if np.argmax(val) == 1 and val[0, 1] > m:  # forward
                        u[i, j], v[i, j] = d
                        m = val[0, 1]
    return [x, y, u, v], c

def derivative(graph):
    d = []
    for i in range(len(graph)-1):
        d.append(graph[i+1]-graph[i])
    return d

def half_index(a):
    m = (a.max() + a.min())/2
    m2 = abs(m - a[0])
    ind = 0
    for i, v in enumerate(a):
        if abs(m - v) < m2:
            m2 = abs(m - v)
            ind = i
    return ind

def create_bernuli_moving_window(n):
    n -= 1
    def factorial(x):
        m = 1.0
        for i in range(1, x + 1):
            m *= i
        return m

    w = []
    for k in range(n+1):
        w.append((factorial(n)/(factorial(k)*factorial(n-k)))*(0.5**n))

    return w

def get_color_map(agent):
    color_map = []
    length = max(sqv.RECT_WIDTH, sqv.RECT_HEIGHT)
    for f in range(length):
        color_map.append([])
        for l in range(length):
            color_map[-1].append([])
            for r in range(length):
                score = agent.q_function.hypot(np.array([[l,f,r]]))[0].reshape(3)
                a = np.linalg.norm(score)
                color_map[-1][-1].append(score/(a if a > 0.0 else 1.0))
        color_map[-1] = np.array(color_map[-1])
    return color_map