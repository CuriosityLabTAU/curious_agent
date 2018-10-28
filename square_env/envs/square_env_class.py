import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

DETECT_COLLISION = True

RECT_WIDTH = 15
RECT_HEIGHT = 15

AGENT_DRAWING_SIZE = 10
DRAWING_RECT_BEGIN = 100

OBSERVATION_SIZE = 3

WINDOW_HEIGHT = 500
WINDOW_WIDTH = 500

INIT_LOCATIONS = [[5, 5], [3, 9], [15, 12], [9, 14], [5, 11]]
INIT_DIRECTIONS = [[1, 0], [0, -1], [0, 0], [0, 0], [0, 0]]


AGENTS_COUNT = 1


def set_global(name, val):
    globals()[name] = val


class SquareEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):

        self.observation_space = spaces.Box(0, max(RECT_WIDTH, RECT_HEIGHT), [3], dtype="int32")
        # agent's distance from each wall where the wall he is not looking at
        # is -30

        self.action_space = spaces.Box(0, 2, [1], dtype="int32")
        # 3 possible action
        # turn left (-1)
        # move forward (0)
        # turn right (1)

        self.square_space = spaces.Box(np.array([0, 0]), np.array([RECT_WIDTH, RECT_HEIGHT]), dtype="int32")

        self.agents = []
        # 2d vectors [x,y],[dx,dy] where dx,dy is the direction in which the agent is looking

        self.viewer = None
        self._seed = 1
        self.agents_render = []
        self.directions_render = []

    def _collides(self, agent):
        for i in self.agents:
            if not i is agent:
                if np.all(agent == i["loc"]):
                    return True
        return False

    def _in_bounds(self, location):
        return self.square_space.contains(location)

    def _get_left(self, agent, dir):
        if dir[0] == -1: # looking left
            l = agent[1]
            for i in self.agents:
                if i["loc"][0] == agent[0] and i["loc"][1] < agent[1]: # same x
                    l = min(l,agent[1] - i["loc"][1])
            return l
        elif dir[0] == 1: # looking right
            l = RECT_HEIGHT-agent[1]
            for i in self.agents:
                if i["loc"][0] == agent[0] and i["loc"][1] > agent[1]: # same x
                    l = min(l, i["loc"][1] - agent[1])
            return l
        elif dir[1] == 1: # looking up
            l = agent[0]
            for i in self.agents:
                if i["loc"][1] == agent[1] and i["loc"][0] < agent[0]: # same y
                    l = min(l, agent[0] - i["loc"][0])
            return l
        l = RECT_WIDTH-agent[0]
        for i in self.agents:
            if i["loc"][1] == agent[1] and i["loc"][0] > agent[0]:  # same x
                l = min(l, i["loc"][0] - agent[0])
        return l

    def _get_front(self, agent, direction):
        if direction[0] == -1:  # looking left
            f = agent[0]
            for i in self.agents:
                if i["loc"][1] == agent[1] and i["loc"][0] < agent[0]:
                    f = min(f, agent[1]-i["loc"][0])
            return f
        elif direction[0] == 1:  # looking right
            f = RECT_WIDTH-agent[0]
            for i in self.agents:
                if i["loc"][1] == agent[1] and i["loc"][0] > agent[0]:
                    f = min(f, i["loc"][0] - agent[0])
            return f
        elif direction[1] == 1:  # looking up
            f = RECT_HEIGHT-agent[1]
            for i in self.agents:
                if i["loc"][0] == agent[0] and i["loc"][1] > agent[1]:
                    f = min(f, i["loc"][1] - agent[1])
            return f
        f = agent[1]
        for i in self.agents:
            if i["loc"][0] == agent[0] and i["loc"][1] < agent[1]:
                f = min(f, agent[1] - i["loc"][1])
        return f

    def _get_right(self, agent, direction):
        if direction[0] == -1:  # looking left
            r = RECT_HEIGHT-agent[1]
            for i in self.agents:
                if i["loc"][0] == agent[0] and i["loc"][1] > agent[1]:
                    r = min(r, i["loc"][1] - agent[1])
            return r
        elif direction[0] == 1:  # looking right
            r = agent[1]
            for i in self.agents:
                if i["loc"][0] == agent[0] and i["loc"][1] < agent[1]:
                    r = min(r, agent[1] - i["loc"][1])
            return r
        elif direction[1] == 1:  # looking up
            r = RECT_WIDTH-agent[0]
            for i in self.agents:
                if i["loc"][1] == agent[1] and i["loc"][1] > agent[1]:
                    r = min(r, i["loc"][1] - agent[1])
            return r
        r = agent[0]
        for i in self.agents:
            if i["loc"][1] == agent[1] and i["loc"][1] < agent[1]:
                r = min(r, agent[1] - i["loc"][1])
        return r

    def _get_observation(self, agent, agent_direction):
        """if self.agent_direction[0] == 1: # looking right
            ob[0] = -10
        else:
            ob[0] = self.agent[0]

        if self.agent_direction[0] == -1: # looking left
            ob[1] = -10
        else:
            ob[1] = 10-self.agent[0]

        if self.agent_direction[1] == 1:  # looking up
            ob[2] = -10
        else:
            ob[2] = self.agent[1]

        if self.agent_direction[1] == -1:  # looking down
            ob[3] = -10
        else:
            ob[3] = 10 - self.agent[1]

        if self.agent_direction[0] == 1:  # looking right
            ob[0] = 10-self.agent[0]
        if self.agent_direction[0] == -1:  # looking left
            ob[0] = self.agent[0]
        if self.agent_direction[1] == 1:  # looking up
            ob[0] = 10 - self.agent[1]
        if self.agent_direction[1] == -1:  # looking down
            ob[0] = self.agent[1]
        """

        return np.array([self._get_left(agent,agent_direction),
                         self._get_front(agent,agent_direction),
                         self._get_right(agent,agent_direction)])

    def _get_info(self):
        a = []
        for i in self.agents:
            a.append({"loc": np.copy(i["loc"]), "dir": np.copy(i["dir"])})
        return a

    def _take_action(self, action, agent):
        if action == -1: # move left
            agent["dir"] = agent["dir"][::-1]
            agent["dir"][0] = -agent["dir"][0]

        elif action == 0:
            if DETECT_COLLISION:
                if self._collides(agent['loc'] + agent['dir']):
                    return
            if self._in_bounds(agent["loc"]+agent["dir"]):
                agent["loc"] += agent["dir"]

        elif action == 1:
            agent["dir"] = agent["dir"][::-1]
            agent["dir"][1] = -agent["dir"][1]

    def step(self, action, index=-1):
        assert self.action_space.contains(action)
        if index == -1:
            action = action[np.arange(len(action)),0]-1
            for i in range(len(self.agents)):
                self._take_action(action[i][0],self.agents[i])

            return self._get_all_observations(),\
                   0,\
                   False,\
                   self._get_info()

        action = action[0] - 1
        self._take_action(action, self.agents[index])
        return self._get_observation(self.agents[index]["loc"],self.agents[index]["dir"]),\
            0,\
            False,\
            self._get_info()

    def _get_all_observations(self):
        l = []
        for i in self.agents:
            l.append(self._get_observation(i["loc"],i["dir"]))
        return l

    def reset(self, render=True):

        self.agents = []
        for i in range(AGENTS_COUNT):
            self.agents.append({"loc": np.array(INIT_LOCATIONS[i]), "dir": np.array(INIT_DIRECTIONS[i])})

        if render:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(WINDOW_WIDTH, WINDOW_HEIGHT)

            self.agents_render = []

            l, r, t, b = -AGENT_DRAWING_SIZE/2, AGENT_DRAWING_SIZE/2, AGENT_DRAWING_SIZE/2, -AGENT_DRAWING_SIZE/2

            for i in range(AGENTS_COUNT):
                agent = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                self.agents_render.append(rendering.Transform())
                agent.add_attr(self.agents_render[i])
                agent.set_color(0, .5, 0)
                self.viewer.add_geom(agent)

            self.directions_render = []

            l, r, t, b = -AGENT_DRAWING_SIZE/5, AGENT_DRAWING_SIZE/5, AGENT_DRAWING_SIZE/5, -AGENT_DRAWING_SIZE/5

            for i in range(AGENTS_COUNT):
                direction = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                direction.set_color(.5, 0, 0)
                self.directions_render.append(rendering.Transform())
                direction.add_attr(self.directions_render[i])
                self.viewer.add_geom(direction)

            l, r, t, b = 0, RECT_WIDTH*AGENT_DRAWING_SIZE, 0, 2

            env_border = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            border_render = rendering.Transform()
            env_border.add_attr(border_render)
            border_render.set_translation(DRAWING_RECT_BEGIN,DRAWING_RECT_BEGIN) # bottom
            self.viewer.add_geom(env_border)

            env_border = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            border_render = rendering.Transform()
            env_border.add_attr(border_render)
            border_render.set_translation(DRAWING_RECT_BEGIN, DRAWING_RECT_BEGIN+RECT_HEIGHT*AGENT_DRAWING_SIZE) #top
            self.viewer.add_geom(env_border)

            l, r, t, b = 0, 2, RECT_HEIGHT*AGENT_DRAWING_SIZE, 0

            env_border = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            border_render = rendering.Transform()
            env_border.add_attr(border_render)
            border_render.set_translation(DRAWING_RECT_BEGIN, DRAWING_RECT_BEGIN)  # left
            self.viewer.add_geom(env_border)

            env_border = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            border_render = rendering.Transform()
            env_border.add_attr(border_render)
            border_render.set_translation(DRAWING_RECT_BEGIN+AGENT_DRAWING_SIZE*RECT_WIDTH, DRAWING_RECT_BEGIN)  # left
            self.viewer.add_geom(env_border)

        return self._get_all_observations()

    def render(self, mode='human', close=False):
        for i in range(len(self.agents_render)):
            self.agents_render[i].set_translation(*(self.agents[i]["loc"]*AGENT_DRAWING_SIZE+DRAWING_RECT_BEGIN))
        for i in range(len(self.directions_render)):
            self.directions_render[i].set_translation(*((self.agents[i]["loc"]+self.agents[i]["dir"])*AGENT_DRAWING_SIZE+DRAWING_RECT_BEGIN))
        return self.viewer.render()

    def close(self):
        self.viewer.close()