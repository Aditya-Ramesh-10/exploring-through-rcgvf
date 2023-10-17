import torch
from agents.ppo_algo_with_ir import PPOAlgoWithIR

class PPOAlgoRND(PPOAlgoWithIR):
    """Uses intrinsic rewards from RND."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, 
                 discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, lr_anneal_frames=1e8,
                 preprocess_obss=None, reshape_reward=None,
                 intrinsic_reward_coef=0.1, record_room_data=False, use_episodic_counts=False,
                 update_predictor=True,
                 target_net=None, predictor=None, predictor_lr=0.001):

        super().__init__(envs, acmodel, device, num_frames_per_proc, 
                         discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, 
                         clip_eps, epochs, batch_size, lr_anneal_frames,
                         preprocess_obss, reshape_reward, intrinsic_reward_coef, 
                         record_room_data, use_episodic_counts,
                         update_predictor)

        self.target_net = target_net
        self.predictor = predictor
        self.predictor_lr = predictor_lr

        self.target_net.to(self.device)
        self.predictor.to(self.device)

        self.target_dim = self.target_net.output_embedding_size

        self.predictor_criterion = torch.nn.MSELoss()
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), 
                                                    self.predictor_lr,
                                                    eps=adam_eps)

        def lr_lambda(epoch):
            return 1 - min(epoch * self.batch_size/self.epochs, self.lr_anneal_frames) / self.lr_anneal_frames
        
        self.predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.predictor_optimizer, 
                                                                     lr_lambda)

        

    def compute_intrinsic_rewards(self, final_obs, kw_dict={}):
        """
        Computes the intrinsic rewards for the given observations using RND
        """
        all_obs = [self.obss[i][j]
                    for i in range(self.num_frames_per_proc)
                    for j in range(self.num_procs)]

        pp_all_obs = self.preprocess_obss(all_obs, device=self.device)

        with torch.no_grad():
            rnd_targets = self.target_net(pp_all_obs)
            predictions = self.predictor(pp_all_obs)

        self.rnd_targets = rnd_targets.reshape(self.num_frames_per_proc, self.num_procs, 
                                               self.target_dim)

        predictions = predictions.reshape(self.num_frames_per_proc, self.num_procs, 
                                               self.target_dim)

        intrinsic_rewards = torch.norm(self.rnd_targets - predictions, dim=2, p=2)
        
        return self.intrinsic_reward_coef * intrinsic_rewards, kw_dict

    def _update_exps(self, kw_dict={}):
        exps = super()._update_exps()
        exps.targets = self.rnd_targets.transpose(0, 1).reshape(-1, self.target_dim)
        return exps

    def get_intrinsic_loss(self, sb, memory=None):
        """Computes training loss for the RND predictor."""
        preds = self.predictor(sb.obs)
        return self.predictor_criterion(preds, sb.targets), None

    def _append_logs_collect(self, logs, kw_dict={}):
        logs["mean_rnd_targets_batch"] = self.rnd_targets.mean().item()
        logs["min_rnd_targets_batch"] = self.rnd_targets.min().item()
        logs["max_rnd_targets_batch"] = self.rnd_targets.max().item()
        logs["std_rnd_targets_batch"] = self.rnd_targets.std().item()
        return logs

    def _append_logs_opt(self, logs):
        logs["learning_rate_predictor"] = self.predictor_scheduler.get_last_lr()[0]
        return logs
