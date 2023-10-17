from agents.ppo_algo_with_ir import PPOAlgoWithIR

class PPOAlgoEpisodicCounts(PPOAlgoWithIR):
    """Uses episodic counts from the simulator to obtain an intrinsic reward signal."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, 
                 discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, lr_anneal_frames=1e8,
                 preprocess_obss=None, reshape_reward=None,
                 intrinsic_reward_coef=0.1, record_room_data=False, use_episodic_counts=True,
                 update_predictor=False):
        
        assert not update_predictor, "This class does not have a predictor."
        assert use_episodic_counts, "This class requires episodic counts."

        super().__init__(envs, acmodel, device, num_frames_per_proc, 
                         discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, 
                         clip_eps, epochs, batch_size, lr_anneal_frames,
                         preprocess_obss, reshape_reward, intrinsic_reward_coef, 
                         record_room_data, use_episodic_counts,
                         update_predictor)

    def compute_intrinsic_rewards(self, final_obs):
        """Intrinsic reward subsequently divided by sqrt of episodic counts"""
        return self.intrinsic_reward_coef

    def _append_logs_collect(self, logs):
        return logs

    def _append_logs_opt(self, logs):
        return logs
