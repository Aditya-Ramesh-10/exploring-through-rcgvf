import gymnasium as gym
import minigrid
import numpy as np
from typing import Callable, Iterable
from copy import deepcopy


def make_env(env_key, seed=None, pano_obs=False):
    env = gym.make(env_key)
    if pano_obs:
        env = PanoramaObservationWrapper(env)
    else:
        env = TimeStepObservationWrapper(env)

    env.reset(seed=seed)

    env = EpisodicCountWrapper(env)
    return env


class EpisodicCountWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(EpisodicCountWrapper, self).__init__(env)

    def state_extraction_key(self):
        return tuple(self.unwrapped.agent_pos)

class TimeStepObservationWrapper(gym.core.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):

        time_step = self.step_count

        return {
            'mission': obs['mission'],
            'image': obs['image'],
            'time_step': time_step
        }


class PanoramaObservationWrapper(gym.core.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation(self, obs):

        time_step = self.step_count

        return {
            'mission': obs['mission'],
            'image': obs['image'],
            'time_step': time_step,
            'panorama_image': self.get_panorama()
        }

    def get_panorama(self):
        # Use a tmp environment, or its internal step counter will increase
        # env = deepcopy(self.gym_env)
        env = deepcopy(self.env)
        dir = env.agent_dir
        while env.agent_dir != 1:
            env.step(1) # Have the agent point at the same direction
        pano = []
        for _ in range(4):
            frame, *_ = env.step(1)
            pano.append(frame['image'])
        # while env.agent_dir != dir:
        #     env.step(1) # Restore direction
        pano = np.array(pano)
        pano = pano.transpose(1, 2, 3, 0)
        return pano.reshape(pano.shape[0], pano.shape[1], -1)


def is_agent_in_room(agent_pos, room_top, room_size):

    t = np.array(room_top)
    b = t + np.array(room_size)

    return (np.all(t < agent_pos) and np.all(b > agent_pos))
