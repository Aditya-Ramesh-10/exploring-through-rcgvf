import torch
from agents.ppo_algo_with_ir import PPOAlgoWithIR

class PPOAlgoRCGVF(PPOAlgoWithIR):
    """Uses intrinsic rewards from RC-GVF."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, 
                 discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, lr_anneal_frames=1e8,
                 preprocess_obss=None, reshape_reward=None,
                 intrinsic_reward_coef=0.1, record_room_data=False, use_episodic_counts=False,
                 update_predictor=True,
                 pseudo_reward_generator=None, predictor=None, predictor_lr=0.001,
                 gvf_discount=0.6, gvf_lambda=0.9):

        super().__init__(envs, acmodel, device, num_frames_per_proc, 
                         discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, adam_eps, 
                         clip_eps, epochs, batch_size, lr_anneal_frames,
                         preprocess_obss, reshape_reward, intrinsic_reward_coef, 
                         record_room_data, use_episodic_counts,
                         update_predictor)

        self.pseudo_reward_generator = pseudo_reward_generator
        self.predictor = predictor
        self.predictor_lr = predictor_lr

        self.gvf_discount = gvf_discount
        self.gvf_lambda = gvf_lambda

        self.pseudo_reward_generator.to(self.device)
        self.predictor.to(self.device)

        self.num_cumulants = self.pseudo_reward_generator.output_embedding_size

        self.predictor_criterion = torch.nn.MSELoss()
        self.predictor_optimizer = torch.optim.Adam(self.predictor.parameters(), 
                                                    self.predictor_lr,
                                                    eps=adam_eps)

        def lr_lambda(epoch):
            return 1 - min(epoch * self.batch_size/self.epochs, self.lr_anneal_frames) / self.lr_anneal_frames
        
        self.predictor_scheduler = torch.optim.lr_scheduler.LambdaLR(self.predictor_optimizer, 
                                                                     lr_lambda)

        shape = (self.num_frames_per_proc, self.num_procs)

        self.predictor_memory = torch.zeros(shape[1],
                                            self.predictor.memory_size,
                                            device=self.device)
        self.predictor_memories = torch.zeros(*shape,
                                                self.predictor.memory_size,
                                                device=self.device)

        self.cumulants = torch.zeros(*shape, self.num_cumulants,
                                     device=self.device)

        self.prev_cumulant = torch.zeros(self.num_procs, self.num_cumulants,
                                         device=self.device)

        self.previous_cumulants = torch.zeros(*shape, self.num_cumulants,
                                              device=self.device)

        self.prev_act = torch.zeros(self.num_procs,
                                    self.acmodel.number_of_actions,
                                    device=self.device)

        self.previous_actions = torch.zeros(*shape,
                                            self.acmodel.number_of_actions,
                                            device=self.device)

        self.predicted_values = torch.zeros(self.predictor.num_heads,
                                            *shape, self.num_cumulants,
                                            device=self.device)


    def _additional_interaction(self, preprocessed_obs, memory=None):
        kw_dict = {}
        with torch.no_grad():
            cumulant = self.pseudo_reward_generator(preprocessed_obs)
            
            if self.predictor.recurrent:
                predicted_value_mult, predictor_memory = self.predictor(preprocessed_obs, self.prev_act,
                                                                        self.prev_cumulant,
                                                                        self.predictor_memory * self.mask.unsqueeze(1),
                                                                        )
            else:
                predicted_value_mult = self.predictor(preprocessed_obs, self.prev_act,
                                                        self.prev_cumulant)

        kw_dict["cumulant"] = cumulant
        kw_dict["predicted_value_mult"] = predicted_value_mult
        kw_dict["predictor_memory"] = predictor_memory
        return kw_dict

    def _update_interaction(self, i, obs, reward, done, dist, value, memory, action, kw_dict):
        super()._update_interaction(i, obs, reward, done, dist, value, memory, action, kw_dict)

        if self.predictor.recurrent:
            self.predictor_memories[i] = self.predictor_memory
            self.predictor_memory = kw_dict["predictor_memory"]

        self.predicted_values[:, i, :, :] = torch.stack(kw_dict["predicted_value_mult"])

        # Updates for the predictor
        self.previous_actions[i] = self.prev_act
        self.prev_act = torch.nn.functional.one_hot(action,
                                                    self.acmodel.number_of_actions).type(torch.float)
        self.previous_cumulants[i] = self.prev_cumulant
        self.prev_cumulant = kw_dict["cumulant"]
        self.cumulants[i] = kw_dict["cumulant"]

    def _final_interaction(self, preprocessed_obs, kw_dict):

        with torch.no_grad():
            if self.predictor.recurrent:
                next_pseudo_value_mult, _ = self.predictor(preprocessed_obs,
                                                           self.prev_act,
                                                           self.prev_cumulant,
                                                           self.predictor_memory * self.mask.unsqueeze(1),
                                                           )
            else:
                next_pseudo_value_mult = self.predictor(preprocessed_obs)

            next_pseudo_value_mult = torch.stack(next_pseudo_value_mult)

        kw_dict["next_pseudo_value_mult"] = next_pseudo_value_mult
        return kw_dict


    def compute_intrinsic_rewards(self, final_obs=None, kw_dict={}):
        """
        Computes the intrinsic rewards for the given observations.
        """

        pseudo_returns_mult = self.compute_value_targets(self.cumulants,
                                                        kw_dict["next_pseudo_value_mult"])

        pred_error = torch.mean(torch.pow((pseudo_returns_mult - self.predicted_values), 2), dim=0)
        var_preds = torch.var(self.predicted_values, dim=0)
        intrinsic_rewards = torch.sum(var_preds * pred_error, dim=-1)

        kw_dict["pseudo_returns_mult"] = pseudo_returns_mult
        kw_dict["prediction_error"] = pred_error
        kw_dict["variance_predictions"] = var_preds
        
        return self.intrinsic_reward_coef * intrinsic_rewards, kw_dict

    def compute_value_targets(self, cumulants, next_pseudo_value_mult):

        pseudo_return_mult = torch.zeros(self.predicted_values.shape,
                                         device=self.device)

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_mask = next_mask.unsqueeze(1).repeat(1, self.num_cumulants)  # to account for multiple cumulants
            next_value = (self.gvf_lambda * pseudo_return_mult[:, i+1] + (1 - self.gvf_lambda) * self.predicted_values[:, i+1]) if i < self.num_frames_per_proc - 1 else next_pseudo_value_mult

            pseudo_return_mult[:, i, :, :] = cumulants[i] + self.gvf_discount * next_mask * next_value

        return pseudo_return_mult

    def _update_exps(self, kw_dict={}):
        exps = super()._update_exps(kw_dict=kw_dict)

        if self.predictor.recurrent:
            exps.predictor_memory = self.predictor_memories.transpose(0, 1).reshape(-1, *self.predictor_memories.shape[2:])

        exps.previous_actions = self.previous_actions.transpose(0, 1).reshape(-1, self.acmodel.number_of_actions)
        exps.previous_cumulants = self.previous_cumulants.transpose(0, 1).reshape(-1, self.num_cumulants)
        # exps.cumulants = self.cumulants.transpose(0, 1).reshape(-1, self.num_cumulants)

        prs_reshaped = kw_dict["pseudo_returns_mult"].transpose(1, 2).reshape(self.predictor.num_heads, -1, self.num_cumulants)
        prs_reshaped = prs_reshaped.transpose(0, 1)
        exps.targets = prs_reshaped
        
        return exps

    def get_intrinsic_loss(self, sb, memory=None):
        """Computes training loss for the RC-GVF predictor."""

        if self.predictor.recurrent:
            predictions, predictor_memory = self.predictor(sb.obs,
                                                           sb.previous_actions,
                                                           sb.previous_cumulants,
                                                           memory * sb.mask)
        else:
            predictions = self.predictor(sb.obs)
            predictor_memory = None

        ## Compute the loss
        targets = sb.targets
        predictor_loss = []
        for h in range(self.predictor.num_heads):
            predictor_loss.append(self.predictor_criterion(predictions[h], targets[:, h, :]))

        intrinsic_loss = sum(predictor_loss) / self.predictor.num_heads

        return intrinsic_loss, predictor_memory

    def _append_logs_collect(self, logs, kw_dict={}):
        logs["mean_cumulants_batch"] = self.cumulants.mean().item()
        logs["min_cumulants_batch"] = self.cumulants.min().item()
        logs["max_cumulants_batch"] = self.cumulants.max().item()
        logs["std_cumulants_batch"] = self.cumulants.std().item()

        logs["mean_pseudo_return_batch"] = kw_dict["pseudo_returns_mult"].mean().item()
        logs["min_pseudo_return_batch"] = kw_dict["pseudo_returns_mult"].min().item()
        logs["max_pseudo_return_batch"] = kw_dict["pseudo_returns_mult"].max().item()
        logs["std_pseudo_return_batch"] = kw_dict["pseudo_returns_mult"].std().item()

        logs["mean_prediction_error_batch"] = kw_dict["prediction_error"].mean().item()
        logs["min_prediction_error_batch"] = kw_dict["prediction_error"].min().item()
        logs["max_prediction_error_batch"] = kw_dict["prediction_error"].max().item()
        logs["std_prediction_error_batch"] = kw_dict["prediction_error"].std().item()

        logs["mean_variance_predictions_batch"] = kw_dict["variance_predictions"].mean().item()
        logs["min_variance_predictions_batch"] = kw_dict["variance_predictions"].min().item()
        logs["max_variance_predictions_batch"] = kw_dict["variance_predictions"].max().item()
        logs["std_variance_predictions_batch"] = kw_dict["variance_predictions"].std().item()

        return logs

    def _append_logs_opt(self, logs):
        logs["learning_rate_predictor"] = self.predictor_scheduler.get_last_lr()[0]
        return logs
