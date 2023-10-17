from abc import ABC, abstractmethod
import torch
import numpy as np
from collections import defaultdict

from torch_ac.format import default_preprocess_obss
from torch_ac.utils import DictList

from mgrid_utils.par_env import ParallelEnv
from mgrid_utils.env import is_agent_in_room

class BaseAlgoWithIR(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward, intrinsic_reward_coef,
                 record_room_data, use_episodic_counts):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodel : torch.Module
            the model
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        intrinsic_reward_coef : float
            the weight of the intrinsic reward in combination with the extrinsic reward
        record_room_data : bool
            whether to record farthest room data
        use_episodic_counts : bool
            whether to use episodic state counts (privileged information) to scale the intrinsic reward
        """

        # Store parameters

        self.env = ParallelEnv(envs)
        self.acmodel = acmodel
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss or default_preprocess_obss
        self.reshape_reward = reshape_reward

        self.intrinsic_reward_coef = intrinsic_reward_coef

        # Diagnostics
        self.record_room_data = record_room_data
        self.rooms_visited_batch = []
        self.farthest_room_visited = 0

        self.use_episodic_counts = use_episodic_counts

        # Control parameters

        assert self.acmodel.recurrent or self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodel

        self.acmodel.to(self.device)
        self.acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs)

        self.obs = self.env.reset()
        self.obss = [None] * (shape[0])
        if self.acmodel.recurrent:
            self.memory = torch.zeros(shape[1], self.acmodel.memory_size, device=self.device)
            self.memories = torch.zeros(*shape, self.acmodel.memory_size, device=self.device)
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(*shape, device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Maintain episodic state counts
        self.episodic_state_count_dict = [defaultdict(lambda: 0) for i in range(self.num_procs)]
        self.episodic_state_counts_t = torch.zeros(self.num_procs, device=self.device)
        self.episodic_state_counts = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_reshaped_return = torch.zeros(self.num_procs, device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [0] * self.num_procs
        self.log_reshaped_return = [0] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            if self.record_room_data:
                self.update_room_info()

            if self.use_episodic_counts:
                # Update episodic state counts
                self.update_episodic_counts()

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                if self.acmodel.recurrent:
                    dist, value, memory = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
                else:
                    dist, value = self.acmodel(preprocessed_obs)

                kw_dict = self._additional_interaction(preprocessed_obs)

            action = dist.sample()

            obs, reward, terminated, truncated, _ = self.env.step(action.cpu().numpy())
            done = tuple(a | b for a, b in zip(terminated, truncated))

            # Update interaction experiences

            self._update_interaction(i, obs, reward, done, dist, value, memory, action, kw_dict)

            # Update log values

            self.log_episode_return += torch.tensor(reward, device=self.device, dtype=torch.float)
            self.log_episode_reshaped_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for i, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(self.log_episode_return[i].item())
                    self.log_reshaped_return.append(self.log_episode_reshaped_return[i].item())
                    self.log_num_frames.append(self.log_episode_num_frames[i].item())

                    # reset episodic state counts dict when done
                    if self.use_episodic_counts:
                        self.episodic_state_count_dict[i] = defaultdict(lambda : 0)

            self.log_episode_return *= self.mask
            self.log_episode_reshaped_return *= self.mask
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            if self.acmodel.recurrent:
                _, next_value, _ = self.acmodel(preprocessed_obs, self.memory * self.mask.unsqueeze(1))
            else:
                _, next_value = self.acmodel(preprocessed_obs)

            kw_dict = self._final_interaction(preprocessed_obs, kw_dict)


        intrinsic_rewards, kw_dict = self.compute_intrinsic_rewards(final_obs=self.obs, 
                                                                    kw_dict=kw_dict)
        if self.use_episodic_counts:
            intrinsic_rewards = intrinsic_rewards/torch.sqrt(self.episodic_state_counts)
        
        self.rewards += intrinsic_rewards
        # clamped rewards
        self.rewards = torch.clamp(self.rewards, -1, 1)

        self._compute_advantages(next_value)

        # Define experiences:
        exps = self._update_exps(kw_dict=kw_dict)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "reshaped_return_per_episode": self.log_reshaped_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames,
            "mean_intrinsic_reward_batch": intrinsic_rewards.mean().item(),
            "min_intrinsic_reward_batch": intrinsic_rewards.min().item(),
            "max_intrinsic_reward_batch": intrinsic_rewards.max().item(),
            "std_intrinsic_reward_batch": intrinsic_rewards.std().item(),
        }

        if self.record_room_data:
            logs["farthest_room_visited_batch"] = np.max(self.rooms_visited_batch)
            logs["farthest_room_visited_overall"] = self.farthest_room_visited
            self.rooms_visited_batch = []

        logs = self._append_logs_collect(logs, kw_dict=kw_dict)

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_reshaped_return = self.log_reshaped_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def _append_logs_collect(self, logs, kw_dict={}):
        pass

    @abstractmethod
    def update_parameters(self):
        pass

    @abstractmethod
    def compute_intrinsic_rewards(self, final_obs=None, kw_dict={}):
        pass

    def _additional_interaction(self, preprocessed_obs, memory=None):
        return {}

    def _final_interaction(self, preprocessed_obs, kw_dict):
        return kw_dict

    def _update_interaction(self, i, obs, reward, done, dist, value, memory, action, kw_dict):

            self.obss[i] = self.obs
            self.obs = obs
            if self.acmodel.recurrent:
                self.memories[i] = self.memory
                self.memory = memory
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = action
            self.values[i] = value
            if self.reshape_reward is not None:
                self.rewards[i] = torch.tensor([
                    self.reshape_reward(obs_, action_, reward_, done_)
                    for obs_, action_, reward_, done_ in zip(obs, action, reward, done)
                ], device=self.device)
            else:
                self.rewards[i] = torch.tensor(reward, device=self.device)
            self.log_probs[i] = dist.log_prob(action)

            # Update episodic state counts
            if self.use_episodic_counts:
                self.episodic_state_counts[i] = self.episodic_state_counts_t

    def _compute_advantages(self, next_value):
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask


    def _update_exps(self, kw_dict={}):
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        if self.acmodel.recurrent:
            # T x P x D -> P x T x D -> (P * T) x D
            exps.memory = self.memories.transpose(0, 1).reshape(-1, *self.memories.shape[2:])
            # T x P -> P x T -> (P * T) x 1
            exps.mask = self.masks.transpose(0, 1).reshape(-1).unsqueeze(1)
        # for all tensors below, T x P -> P x T -> P * T
        exps.action = self.actions.transpose(0, 1).reshape(-1)
        exps.value = self.values.transpose(0, 1).reshape(-1)
        exps.reward = self.rewards.transpose(0, 1).reshape(-1)
        exps.advantage = self.advantages.transpose(0, 1).reshape(-1)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 1).reshape(-1)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        return exps

    def update_episodic_counts(self):
        agent_poses = self.env.state_extraction_key()
        for j, pose_j in enumerate(agent_poses):            
            self.episodic_state_count_dict[j][pose_j] += 1
            self.episodic_state_counts_t[j] = self.episodic_state_count_dict[j][pose_j]

    def update_room_info(self):
        agent_poses = self.env.state_extraction_key()
        rooms_all_envs = self.env.get_rooms()
        for k in range(len(agent_poses)):
            for room_num in range(len(rooms_all_envs[k])):
                in_room = is_agent_in_room(agent_poses[k],
                                           rooms_all_envs[k][room_num].top,
                                           rooms_all_envs[k][room_num].size)

                if in_room:
                    self.rooms_visited_batch.append(room_num)
                    if room_num > self.farthest_room_visited:
                        self.farthest_room_visited = room_num
                    break