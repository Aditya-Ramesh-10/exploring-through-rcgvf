from .agent import *
from .env import *
from .format import *
from .other import *
from .storage import *

from gymnasium.envs.registration import register

register(
    id="MiniGrid-MultiRoom-N7-S8-v0",
    entry_point="minigrid.envs:MultiRoomEnv",
    kwargs={"minNumRooms": 7, "maxNumRooms": 7, "maxRoomSize": 8},
)

register(
    id="MiniGrid-MultiRoom-N12-S10-v0",
    entry_point="minigrid.envs:MultiRoomEnv",
    kwargs={"minNumRooms": 12, "maxNumRooms": 12, "maxRoomSize": 10},
)