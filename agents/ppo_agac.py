import torch
from torch.distributions import kl_divergence, Categorical
from agents.ppo_algo_with_ir import PPOAlgoWithIR

from collections import defaultdict

class PPOAlgoAGAC(PPOAlgoWithIR):
    """Uses intrinsic rewards from AGAC."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, 
                 discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, lr_anneal_frames=1e8,
                 preprocess_obss=None, reshape_reward=None,
                 intrinsic_reward_coef=0.1, record_room_data=False, use_episodic_counts=True,
                 update_predictor=True, predictor=None, predictor_lr=0.001, 
                 episodic_count_coef=0.01):

        super().__init__(envs, acmodel, device, num_frames_per_proc, 
                         discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, 
                         clip_eps, epochs, batch_size, lr_anneal_frames,
                         preprocess_obss, reshape_reward, intrinsic_reward_coef, 
                         record_room_data, use_episodic_counts,
                         update_predictor)

        self.predictor = predictor
        self.predictor_lr = predictor_lr

        self.episodic_count_coef = episodic_count_coef

        self.predictor.to(self.device)

        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), 
                                                    self.predictor_lr,
                                                    eps=adam_eps)

        def lr_lambda(epoch):
            return 1 - min(epoch * self.batch_size/self.epochs, self.lr_anneal_frames) / self.lr_anneal_frames
        
        self.predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.predictor_optimizer, 
                                                                     lr_lambda)

        ## AGAC specific

        shape = (self.num_frames_per_proc, self.num_procs)

        self.intrinsic_values = torch.zeros(*shape, device=self.device)
        self.intrinsic_advantages = torch.zeros(*shape, device=self.device)

        if self.predictor.recurrent:
            self.predictor_memory = torch.zeros(shape[1],
                                                self.predictor.memory_size,
                                                device=self.device)
            self.predictor_memories = torch.zeros(*shape,
                                                  self.predictor.memory_size,
                                                  device=self.device)

        self.predictor_log_probs = torch.zeros(*shape, device=self.device)

        self.target_action_logits = torch.zeros(*shape,
                                                self.predictor.number_of_actions,
                                                device=self.device)


    def _additional_interaction(self, preprocessed_obs, memory=None):
        kw_dict = {}
        with torch.no_grad():
            if self.predictor.recurrent:
                predicted_dist, predictor_memory = self.predictor(preprocessed_obs,
                                                                    self.predictor_memory * self.mask.unsqueeze(1))
            else:
                predicted_dist = self.predictor(preprocessed_obs)

        kw_dict["predicted_dist"] = predicted_dist
        kw_dict["predictor_memory"] = predictor_memory
        return kw_dict

    def _update_interaction(self, i, obs, reward, done, dist, value, memory, action, kw_dict):
        super()._update_interaction(i, obs, reward, done, dist, value, memory, action, kw_dict)

        if self.predictor.recurrent:
            self.predictor_memories[i] = self.predictor_memory
            self.predictor_memory = kw_dict["predictor_memory"]

        self.predictor_log_probs[i] = kw_dict["predicted_dist"].log_prob(action)

        self.target_action_logits[i] = dist.logits
        self.intrinsic_advantages[i] = dist.log_prob(action) - kw_dict["predicted_dist"].log_prob(action)
        self.intrinsic_values[i] = kl_divergence(dist, kw_dict["predicted_dist"])

    def _compute_advantages(self, next_value):
        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_value
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            delta = self.rewards[i] + self.discount * next_value * next_mask - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask

        # Add intrinsic advantages and values to experiences
        self.advantages = self.advantages + self.intrinsic_reward_coef * self.intrinsic_advantages
        self.values = self.values + self.intrinsic_reward_coef * self.intrinsic_values

    def compute_intrinsic_rewards(self, final_obs, kw_dict={}):
        """
        Computes the intrinsic rewards based on Episodic counts.
        The adversarial bonus is directly added to advantages.
        """
        return self.episodic_count_coef, kw_dict

    def _update_exps(self, kw_dict={}):
        exps = super()._update_exps()
        exps.target_dist_logits = self.target_action_logits.transpose(0, 1).reshape(-1, self.predictor.number_of_actions)

        if self.predictor.recurrent:
            exps.predictor_memory = self.predictor_memories.transpose(0, 1).reshape(-1, *self.predictor_memories.shape[2:])      
        
        return exps

    def get_intrinsic_loss(self, sb, memory=None):
        """Computes training loss for the policy predictor."""
        predictor_memory = None

        targets = Categorical(logits=sb.target_dist_logits)

        if self.predictor.recurrent:
            predicted_dists, predictor_memory = self.predictor(sb.obs, memory * sb.mask)
        else:
            predicted_dists = self.predictor(sb.obs)

        predicted_dists.probs = predicted_dists.probs + 1e-8
        predictor_loss = kl_divergence(targets, predicted_dists)
        intrinsic_loss = predictor_loss.mean()

        return intrinsic_loss, predictor_memory

    def _append_logs_collect(self, logs, kw_dict={}):
        logs["mean_intrinsic_values_batch"] = self.intrinsic_values.mean().item()
        logs["min_intrinsic_values_batch"] = self.intrinsic_values.min().item()
        logs["max_intrinsic_values_batch"] = self.intrinsic_values.max().item()
        logs["std_intrinsic_values_batch"] = self.intrinsic_values.std().item()
        return logs

    def _append_logs_opt(self, logs):
        logs["learning_rate_predictor"] = self.predictor_scheduler.get_last_lr()[0]
        return logs
