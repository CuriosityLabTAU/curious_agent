import numpy as np
import square_env.envs as sqv
import curious_agent as cru

class MovingCube(cru.CuriousAgent):
    def __init__(self, index):
        cru.CuriousAgent.__init__(self, index)
        self.turning = False

    def take_step(self, env, state, prev_error):
        if self.turning:
            self.turning = False
            env.step(np.array([0]), index=self.index)
            return range(6)
        location = env.agents[self.index]['loc']
        direction = env.agents[self.index]['dir']
        turn = np.argmax(abs(direction))
        tmax = sqv.RECT_HEIGHT if turn else sqv.RECT_WIDTH
        if (location[turn] == tmax and direction[turn] == 1) or (location[turn] == 0 and direction[turn] == -1):
            self.turning = True
            env.step(np.array([0]), index=self.index)
            return range(6)
        env.step(np.array([1]), index=self.index)
        return range(6)


