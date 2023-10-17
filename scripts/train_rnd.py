import time
import datetime
import torch
import numpy as np
import torch_ac
import sys
import mgrid_utils as utils
from model import ACModel, RandomFeatureNetwork

from agents.ppo_rnd import PPOAlgoRND

import wandb
project_name = 'rcgvf_minigrid'

config_defaults = dict(
    # Environment details -----------------------------------------------------
    env='MiniGrid-KeyCorridorS4R3-v0',
    pano_obs=False, #policy uses non-panoramic observations
    seed=636,  # for both environment and algorithm
    # Logging details ---------------------------------------------------------
    log_interval=5,
    save_interval=0,
    # Gif recording -----------------------------------------------------------
    record_gif_interval=0,
    n_episodes_recording=2,
    greedy_action_recording=False,
    # Algo details ------------------------------------------------------------
    algo='ppo-rnd',
    # Base algo -----------------------
    procs=16,
    frames=3e7,
    lr_anneal_frames=3e7,
    frames_per_proc=128,
    batch_size=256,
    epochs=4,
    clip_eps=0.2,
    discount=0.99,
    lr=0.0002,
    gae_lambda=0.95,
    entropy_coef=1e-5,
    value_loss_coef=0.5,
    max_grad_norm=0.5,
    optim_eps=1e-8,
    optim_alpha=0.99,
    recurrence=4,
    text=False,
    # RND specific -----------------------------
    rnd_target_embedding_size=128,
    intrinsic_coef=0.0005,
    lr_predictor_factor=0.75,
    record_room_data=False,
    use_episodic_counts=False,
)

def main():
    wandb.init(config=config_defaults, project=project_name)
    args = wandb.config
    args.mem = args.recurrence > 1

    hash_config = str(hash(frozenset(args.items())))

    date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
    default_model_name = f"{args.env}_{args.algo}_seed{args.seed}_{date}_{hash_config}"

    model_name = default_model_name
    model_dir = utils.get_model_dir(model_name)

    # Load loggers and Tensorboard writer

    txt_logger = utils.get_txt_logger(model_dir)
    csv_file, csv_logger = utils.get_csv_logger(model_dir)
    
    # Log command and all script arguments
    txt_logger.info("{}\n".format(" ".join(sys.argv)))
    txt_logger.info("{}\n".format(args))

    # Set seed for all randomness sources
    utils.seed(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt_logger.info(f"Device: {device}\n")

    # Load environments

    envs = []

    for i in range(args.procs):

        envs.append(utils.make_env(args.env, args.seed + 10000 * i,
                    pano_obs=args.pano_obs))

    for e in envs:
        o = e.reset()
        print(e.agent_pos)
    txt_logger.info("Environments loaded\n")

    if args.record_gif_interval > 0:
        recording_env = utils.make_env(args.env, args.seed,
                                        pano_obs=args.pano_obs,
                                        )
        
    try:
        status = utils.get_status(model_dir)
    except OSError:
        status = {"num_frames": 0, "update": 0}
    txt_logger.info("Training status loaded\n")

    # Load observations preprocessor
    if args.pano_obs:
        obs_space, preprocess_obss = utils.get_pano_obss_preprocessor(envs[0].observation_space)
    else:
        obs_space, preprocess_obss = utils.get_obss_preprocessor(envs[0].observation_space)
    if "vocab" in status:
        preprocess_obss.vocab.load_vocab(status["vocab"])
    txt_logger.info("Observations preprocessor loaded")

    # Load model
    acmodel = ACModel(obs_space, envs[0].action_space, args.mem, args.text)

    random_target_network = RandomFeatureNetwork(obs_space,
                                                 output_embedding_size=args.rnd_target_embedding_size,
                                                 pano_obs=args.pano_obs)

    predictor = RandomFeatureNetwork(obs_space,
                                     output_embedding_size=args.rnd_target_embedding_size,
                                     pano_obs=args.pano_obs)

    if "model_state" in status:
        acmodel.load_state_dict(status["model_state"])
    acmodel.to(device)
    txt_logger.info("Model loaded\n")
    txt_logger.info("{}\n".format(acmodel))
    wandb.log({"model details": "{}\n".format(acmodel)})

    if args.algo == "ppo-rnd":
        algo = PPOAlgoRND(envs, acmodel, device, args.frames_per_proc, args.discount,
                          args.lr, args.gae_lambda, args.entropy_coef, args.value_loss_coef, 
                          args.max_grad_norm, args.recurrence, args.optim_eps, 
                          args.clip_eps, args.epochs, args.batch_size, args.lr_anneal_frames,
                          preprocess_obss, intrinsic_reward_coef=args.intrinsic_coef,
                          record_room_data=args.record_room_data, use_episodic_counts=args.use_episodic_counts,
                          target_net=random_target_network, predictor=predictor, 
                          predictor_lr=args.lr * args.lr_predictor_factor)
        
    else:
        raise ValueError("Incorrect algorithm name: {}".format(args.algo))
    

    if "optimizer_state" in status:
        algo.optimizer.load_state_dict(status["optimizer_state"])
    txt_logger.info("Optimizer loaded\n")

    # Train model

    num_frames = status["num_frames"]
    update = status["update"]
    start_time = time.time()

    while num_frames < args.frames:
        # Update model parameters

        update_start_time = time.time()
        exps, logs1 = algo.collect_experiences()
        logs2 = algo.update_parameters(exps)
        logs = {**logs1, **logs2}
        update_end_time = time.time()

        num_frames += logs["num_frames"]
        update += 1

        # Print logs

        if update % args.log_interval == 0:
            fps = logs["num_frames"]/(update_end_time - update_start_time)
            duration = int(time.time() - start_time)
            return_per_episode = utils.synthesize(logs["return_per_episode"])
            rreturn_per_episode = utils.synthesize(logs["reshaped_return_per_episode"])
            num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

            header = ["update", "frames", "FPS", "duration"]
            data = [update, num_frames, fps, duration]
            header += ["rreturn_" + key for key in rreturn_per_episode.keys()]
            data += rreturn_per_episode.values()
            header += ["num_frames_" + key for key in num_frames_per_episode.keys()]
            data += num_frames_per_episode.values()
            header += ["entropy", "value", "policy_loss", "value_loss", "grad_norm"]
            data += [logs["entropy"], logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"]]

            txt_logger.info(
                "U {} | F {:06} | FPS {:04.0f} | D {} | rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {} | H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f}"
                .format(*data))

            header += ["return_" + key for key in return_per_episode.keys()]
            data += return_per_episode.values()

            if 'rnd' in args.algo:

                header += ["learning_rate"]
                data += [logs["learning_rate"]]

                header += ["learning_rate_predictor"]
                data += [logs["learning_rate_predictor"]]

                header += ["predictor_grad_norm", "predictor_loss"]
                data += [logs["predictor_grad_norm"], logs["predictor_loss"]]

                header += ["mean_intrinsic_reward_batch", "min_intrinsic_reward_batch",
                          "max_intrinsic_reward_batch", "std_intrinsic_reward_batch"]

                data += [logs["mean_intrinsic_reward_batch"], logs["min_intrinsic_reward_batch"],
                        logs["max_intrinsic_reward_batch"], logs["std_intrinsic_reward_batch"]]

                header += ["mean_rnd_targets_batch", "min_rnd_targets_batch",
                          "max_rnd_targets_batch", "std_rnd_targets_batch"]

                data += [logs["mean_rnd_targets_batch"], logs["min_rnd_targets_batch"],
                        logs["max_rnd_targets_batch"], logs["std_rnd_targets_batch"]]

                if args.record_room_data:

                    header += ["farthest_room_in_batch"]
                    data += [logs["farthest_room_visited_batch"]]

                    header += ["farthest_room_ever"]
                    data += [logs["farthest_room_visited_overall"]]

            if status["num_frames"] == 0:
                csv_logger.writerow(header)
            csv_logger.writerow(data)
            csv_file.flush()

            log_dict = {}
            for field, value in zip(header, data):
                log_dict[field] = value

            wandb.log(log_dict)
        # Save status

        if args.save_interval > 0 and update % args.save_interval == 0:
            status = {"num_frames": num_frames, "update": update,
                      "model_state": acmodel.state_dict(),
                      "optimizer_state": algo.optimizer.state_dict()}
            if hasattr(preprocess_obss, "vocab"):
                status["vocab"] = preprocess_obss.vocab.vocab
            utils.save_status(status, model_dir)
            txt_logger.info("Status saved")

        if args.record_gif_interval > 0 and update % args.record_gif_interval == 0:

            r_obs = recording_env.reset()
            print(recording_env.agent_pos)

            r_memory = torch.zeros(1, acmodel.memory_size,
                                 device=device)
            r_data = []
            r_ep_dict = {}
            r_ep_num = 0
            r_ep_return = 0

            r_obs_list = []
            r_action_list = []
            r_reward_list = []

            r_frames = []

            # Create a window to view the environment
            # env.render('human')

            while r_ep_num < args.n_episodes_recording:

                r_frames.append(np.moveaxis(recording_env.render("rgb_array"),
                                            2, 0))

                r_obs_list.append(r_obs["image"])
                r_preprocessed_obs = preprocess_obss([r_obs], device=device)

                r_dist, r_value, r_memory = acmodel(r_preprocessed_obs, r_memory)

                if args.greedy_action_recording:
                    r_action = r_dist.probs.max(1, keepdim=True)[1]
                else:
                    r_action = r_dist.sample()

                r_action_list.append(r_action.cpu().numpy())

                r_obs, r_reward, r_done, _ = recording_env.step(r_action.cpu().numpy())

                r_ep_return += r_reward

                r_reward_list.append(r_reward)

                if r_done:

                    r_ep_dict['features'] = r_obs_list
                    r_ep_dict['actions'] = np.array(r_action_list).reshape(-1)
                    r_ep_dict['rewards'] = r_reward_list
                    r_ep_dict['ep_return'] = r_ep_return

                    r_data.append(r_ep_dict)
                    print("Episode: ", r_ep_num)
                    r_ep_dict = {}
                    r_ep_num += 1
                    r_obs = recording_env.reset()
                    r_memory = torch.zeros(1, acmodel.memory_size,
                                         device=device)

                    r_obs_list = []
                    r_action_list = []
                    r_reward_list = []
                    r_ep_return = 0

            wandb.log({"recording": wandb.Video(np.array(r_frames),
                                                fps=10, format="gif"),
                       "frames": num_frames})


if __name__ == '__main__':
    main()
