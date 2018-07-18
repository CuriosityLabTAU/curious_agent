import numpy as np
from neural_network import NeuralNetwork
from recurrent_neural_network import RecurrentNeuralNetwork
import square_env.envs as sqv
from curious_agent import *


class RandomAgent(CuriousAgent):
    def take_step(self, env, state, prev_error):

        # step 1: get state from env (achieved by input)

        # step 2: choose a random action

        action = np.zeros(3)
        max_ind = env.action_space.sample()[0]
        action[max_ind] = 1

        # step 3 achieved

        # step 4: perform an action

        observation, _, _, info = env.step(self._unravel_action(action),self.index)

        # step 4 achieved

        # step 5: calculate error

        predicted_state_by_action = self.learner.hypot(np.array([np.concatenate((state, action))]))[0]

        error = predicted_state_by_action-observation

        error = np.sqrt(error.dot(error))

        #error = np.log(np.abs(error)+1)

        delta_error = error-prev_error

        # step 5 achieved

        # step 6: update values and return from function

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
            self.learner.iteration(np.array([np.concatenate((state, action))]),
                                   np.array([observation]), alpha=self.learner_alpha)

        self._set_learner_alpha()
        self._set_gamma()

        return observation, error, info, 0, delta_error, predicted_state_by_action
