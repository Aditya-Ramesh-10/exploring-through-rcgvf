from multiprocessing import Process, Pipe
import gymnasium as gym


def worker(conn, env):
    while True:
        cmd, data = conn.recv()
        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(data)
            if terminated or truncated:
                obs, _ = env.reset()
            conn.send((obs, reward, terminated, truncated, info))
        elif cmd == "reset":
            obs, _ = env.reset()
            conn.send(obs)
        elif cmd == 'state_extraction_key':
            agent_pos = env.state_extraction_key()
            conn.send(agent_pos)
        elif cmd == 'pose_extraction':
            agent_pose = env.pose_extraction()
            conn.send(agent_pose)
        elif cmd == 'get_rooms':
            rooms = env.rooms
            conn.send(rooms)
        else:
            raise NotImplementedError


class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(self, envs):
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()[0]] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, terminated, truncated, info = self.envs[0].step(actions[0])
        if terminated or truncated:
            obs, _ = self.envs[0].reset()
        results = zip(*[(obs, reward, terminated, truncated, info)] + [local.recv() for local in self.locals])
        return results

    def state_extraction_key(self):
        for local in self.locals:
            local.send(("state_extraction_key", None))
        results = [self.envs[0].state_extraction_key()] + [local.recv() for local in self.locals]
        return results

    def pose_extraction(self):
        for local in self.locals:
            local.send(("pose_extraction", None))
        results = [self.envs[0].pose_extraction()] + [local.recv() for local in self.locals]
        return results

    def get_rooms(self):
        for local in self.locals:
            local.send(("get_rooms", None))
        results = [self.envs[0].rooms] + [local.recv() for local in self.locals]
        return results

    def render(self):
        raise NotImplementedError
