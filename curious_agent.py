import numpy as np
from neural_network import NeuralNetwork
from recurrent_neural_network import RecurrentNeuralNetwork
import square_env.envs as sqv


def set_global(name, val):
    """
    sets the value of a global variable
    :param name: string, the name of the global variable
    :param val: the new value
    """
    globals()[name] = val

IS_AGENT_RECURRENT = False
# a boolean indicating whether or not the agent's neural network is recurrent

MIN_EPSILON = 0.0005
# the minimum possible probability of taking a random action

ALL_ACTIONS = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
# an array of all the possible actions to take

AGENT_GAMMA = 0.9
# the init gamma variable of the agent

AGENT_LEARNER_ALPHA = 0.001
# the learning rate of the agent's learner

AGENT_LEARNER_NETWORK_SHAPE = (sqv.OBSERVATION_SIZE + 3, 10, sqv.OBSERVATION_SIZE)
# a tuple of the agent learner's neural network's layers sizes if it is not recurrent

AGENT_LEARNER_NETWORK_SHAPE_RECURRENT = RecurrentNeuralNetwork.create_layers(sqv.OBSERVATION_SIZE + 3, 10, sqv.OBSERVATION_SIZE)
# a tuple of the agent learner's neural network's layers sizes if it is recurrent

AGENT_INIT_COUNTER = 5
# amount of steps to take before performing gradient decent on recurrent network

AGENT_Q_ALPHA = 0.0001
# the learning rate of the agent's q function

AGENT_INIT_EPSILON = 0.1
# initial probability of taking a random action

AGENT_Q_NETWORK = (sqv.OBSERVATION_SIZE, 16, 32, 3)

def relu(x, derivative=False):
    if derivative:
        e = np.exp(x)
        return (e-1)/e  # derivative (logistic function) using the output of relu
    return np.log(np.exp(x)+1)


def linear(x, derivative=False):
    if derivative:
        return 1
    return x


linear = np.vectorize(linear)


def sigmoid(x, derivative=False):
    if derivative:
        return x*(1-x)
    return 1/(1+np.exp(-x))


class CuriousAgent:
    def __init__(self, index):
        self.q_function = NeuralNetwork(AGENT_Q_NETWORK, relu)
        # input -> state
        # output -> value(state)

        self.learner = None
        self.reset_network()

        self.memory = []
        # memory (begins empty)

        self.gamma = AGENT_GAMMA
        self.learner_alpha = AGENT_LEARNER_ALPHA
        self.q_alpha = AGENT_Q_ALPHA
        self.epsilon = AGENT_INIT_EPSILON
        # hyper parameters

        self.index = index
        # index of the agent in the enviroment's agents array

        self.counter = AGENT_INIT_COUNTER

    def reset_network(self):
        if IS_AGENT_RECURRENT:
            self.learner = RecurrentNeuralNetwork(AGENT_LEARNER_NETWORK_SHAPE_RECURRENT, relu)
            self.learner.clear_states()
        else:
            self.learner = NeuralNetwork(AGENT_LEARNER_NETWORK_SHAPE, relu)
        # input -> [state,action]
        # output -> next state
        self.learner_alpha = AGENT_LEARNER_ALPHA

    def _load_minibatch(self, batch_size):
        batch_size = min(len(self.memory),batch_size)
        batch = []
        indexes_used = []

        while batch_size>0:
            index = np.random.randint(len(self.memory))
            if not index in indexes_used:
                indexes_used.append(index)
                batch.append(self.memory[index])
                batch_size -= 1

        return batch

    def remember(self, obj):
        if (not IS_AGENT_RECURRENT) and len(self.memory) > len(self.memory):
            rand = len(self.memory)-int(np.exp(-np.random.rand())*len(self.memory))
            del self.memory[rand]
        self.memory.append(obj)

    def _set_epsilon(self):
        self.epsilon = 1.01 ** (MIN_EPSILON - self.epsilon) * self.epsilon

    def _set_learner_alpha(self):
        MIN_ALPHA = 0.00001
        if self.learner_alpha <= MIN_ALPHA:
            self.learner_alpha = MIN_ALPHA
            return
        self.learner_alpha = 1.0001 ** (self.learner_alpha-1) * self.learner_alpha

    def _set_q_alpha(self):
        MIN_ALPHA = 0.00001
        if self.q_alpha <= MIN_ALPHA:
            self.q_alpha = MIN_ALPHA
            return
        self.q_alpha = 1.0001 ** (self.q_alpha - 1) * self.q_alpha

    def _set_gamma(self):
        pass # leaving gamma the same for now

    def _unravel_action(self, action):
        return np.array([np.argmax(action)])

    def take_step(self, env, state, prev_error):

        # step 1: get state from env (achieved by input)

        # step 2: get the maximum Q value

        q_label = self.q_function.hypot(state)

        max_ind = np.argmax(q_label)

        for i in q_label:
            if np.isnan(i):
                raise Exception("nan occurred")

        # step 2 achieved

        # step 3: get the action to perform using e-greedy

        if np.random.rand() < self.epsilon:
            action = np.zeros(3)
            max_ind = env.action_space.sample()[0]
            action[max_ind] = 1
        else:
            action = ALL_ACTIONS[max_ind]

        # step 3 achieved

        # step 4: perform an action

        observation, _, _, info = env.step(self._unravel_action(action),self.index)

        # step 4 achieved

        # step 5: calculate reward (delta square error)

        predicted_state_by_action = self.learner.hypot(np.array([np.concatenate((state, action))]))[0]

        error = predicted_state_by_action-observation

        error = np.sqrt(error.dot(error))

        #error = np.log(np.abs(error)+1)

        delta_error = error-prev_error

        # step 5 achieved

        # step 6: remember new parameters

        self.remember([state, action, error, observation])

        # step 6 achieved

        # step 7: update values and return from function

        q_tag = error + self.gamma*np.amax(self.q_function.hypot(observation))

        td = (q_label[max_ind]-q_tag)**2

        q_label[max_ind] = q_tag
        self.q_function.iteration(np.array([state]), q_label, alpha=self.q_alpha)
        if IS_AGENT_RECURRENT:
            if self.counter > 0:
                self.counter -= 1
            else:
                self.counter = AGENT_INIT_COUNTER
                inp, label = self._load_batch_for_learner(self.memory)
                self.learner.iteration(inp, label, alpha=self.learner_alpha, step=AGENT_INIT_COUNTER)
                self.learner.clear_states()
                self.memory = []
        else:
            self.learner.iteration(np.array([np.concatenate((state, action))]), np.array([observation]),
                               alpha=self.learner_alpha)

        self._set_learner_alpha()
        self._set_q_alpha()
        self._set_epsilon()
        self._set_gamma()

        return observation, error, info, np.sqrt(td), delta_error, predicted_state_by_action

    def _load_batch_for_learner(self, batch):
        input, output = [], []

        for i in batch:
            input.append(np.concatenate((i[0],i[1])))
            output.append(i[3])

        return np.array(input),np.array(output)

    def _load_batch_for_value(self, batch):
        input,output = [],[]

        for i in batch:
            input.append(i[0])
            output.append(i[2]+self.gamma*self.value_function.hypot(i[3]))

        return np.array(input), np.array(output)

    def train(self, batch_size):
        batch = self._load_minibatch(batch_size)
        learner_input,learner_output = self._load_batch_for_learner(batch)
        value_input,value_output = self._load_batch_for_value(batch)

        self.learner.iteration(learner_input,learner_output,alpha=self.learner_alpha)
        #self.value_function.iteration(value_input,value_output,alpha=self.value_alpha)

        return self.learner.cost(learner_input,learner_output)/min(batch_size,len(self.memory))#,self.value_function.cost(value_input,value_output)