import numpy as np
from neural_network import neural_network
import square_env.envs as sqv

def relu(x,deriv=False):
    if deriv:
        return (np.exp(x)-1)/np.exp(x) # derivative (logistic function) using the output of relu
    return np.log(np.exp(x)+1)

def linear(x,deriv=False):
    if deriv:
        return 1
    return x

def sigmoid(x,deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

def lognonlin(x,deriv=False):
    if deriv:
        return np.exp(x)**(-np.sign(x))
    return np.log(np.abs(x)+1)*np.sign(x)

class CuriousAgent():
    def __init__(self,index, gamma, learner_alpha, q_alpha, epsilon):
        self.q_function = neural_network((sqv.OBSERVATION_SIZE,12,3),(relu,)*2)
        # input -> state
        # output -> value(state)

        self.learner = neural_network((sqv.OBSERVATION_SIZE+3,6,sqv.OBSERVATION_SIZE),(relu,)*2)
        # input -> [state,action]
        # output -> next state

        self.memory = []
        # memory (begins empty)

        self.gamma = gamma
        self.learner_alpha = learner_alpha
        self.q_alpha = q_alpha
        self.epsilon = epsilon
        # hyper parameters

        self.index = index
        # index of the agent in the enviroment's agents array

    def reset_network(self):
        self.learner = neural_network((sqv.OBSERVATION_SIZE + 3, 60, 60, 60, sqv.OBSERVATION_SIZE), (relu,) * 4)
        # input -> [state,action]
        # output -> next state
        self.learner_alpha = 0.0005

    def _load_minibatch(self,batch_size):
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

    def remember(self,obj):
        if len(self.memory) > len(self.memory):
            rand = len(self.memory)-int(np.exp(-np.random.rand())*len(self.memory))
            del self.memory[rand]
        self.memory.append(obj)

    def _set_epsilon(self):
        MIN_EPSILON = 0.0005
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

    def _unravel_action(self,action):
        return np.array([np.argmax(action)])

    def take_step(self,env,state,prev_error):

        # step 1: get state from env (achieved by input)

        # step 2: get the maximum Q value

        ALL_ACTIONS = np.array([[1,0,0],[0,1,0],[0,0,1]])

        q_label = self.q_function.hypot(state)

        max_ind = np.argmax(q_label)

        for i in q_label:
            if np.isnan(i):
                raise Exception("nan occured")

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

        predicted_state_by_action = self.learner.hypot(np.array([np.concatenate((state,action))]))[0]

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
        self.q_function.iteration(np.array([state]),q_label,alpha=self.q_alpha)
        self.learner.iteration(np.array([np.concatenate((state,action))]),np.array([observation]),alpha=self.learner_alpha)


        self._set_learner_alpha()
        self._set_q_alpha()
        self._set_epsilon()
        self._set_gamma()


        return observation,error,info,np.sqrt(td),delta_error,predicted_state_by_action

    def _load_batch_for_learner(self,batch):
        input,output = [],[]

        for i in batch:
            input.append(np.concatenate((i[0],i[1])))
            output.append(i[3])

        return np.array(input),np.array(output)

    def _load_batch_for_value(self,batch):
        input,output = [],[]

        for i in batch:
            input.append(i[0])
            output.append(i[2]+self.gamma*self.value_function.hypot(i[3]))

        return np.array(input), np.array(output)

    def train(self,batch_size):
        batch = self._load_minibatch(batch_size)
        learner_input,learner_output = self._load_batch_for_learner(batch)
        value_input,value_output = self._load_batch_for_value(batch)

        self.learner.iteration(learner_input,learner_output,alpha=self.learner_alpha)
        #self.value_function.iteration(value_input,value_output,alpha=self.value_alpha)

        return self.learner.cost(learner_input,learner_output)/min(batch_size,len(self.memory))#,self.value_function.cost(value_input,value_output)