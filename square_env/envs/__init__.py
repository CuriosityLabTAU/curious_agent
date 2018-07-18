from square_env.envs.square_env_class import SquareEnv, set_global
from square_env.envs.square_env_class import RECT_WIDTH, RECT_HEIGHT, DRAWING_RECT_BEGIN, AGENT_DRAWING_SIZE, INIT_DIRECTIONS, OBSERVATION_SIZE, WINDOW_HEIGHT,WINDOW_WIDTH,INIT_LOCATIONS,AGENTS_COUNT
set_global("AGENTS_COUNT", min(AGENTS_COUNT, len(INIT_LOCATIONS)))
