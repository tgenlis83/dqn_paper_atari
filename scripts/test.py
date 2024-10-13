import glob

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium.wrappers import (
    AtariPreprocessing,
    AutoResetWrapper,
    FrameStack,
    RecordEpisodeStatistics,
    RecordVideo,
)
from matplotlib import pyplot as plt
from networks.dqn import DeepQNetwork
from networks.rainbowdqn import RainbowDeepQNetwork
from tqdm import tqdm
from utils.utils import load_config, select_device, set_seed, test_parse_args


def make_env(env_id, render_mode=None, seed=None, video_folder=None):
    def thunk():
        env = gym.make(
            env_id, render_mode=render_mode, frameskip=1, max_episode_steps=108_000
        )
        env = AtariPreprocessing(
            env, frame_skip=4, terminal_on_life_loss=False, noop_max=30
        )
        env = FrameStack(env, 4)
        env = RecordEpisodeStatistics(env)
        env = AutoResetWrapper(env)
        if video_folder:
            env = RecordVideo(env, video_folder=video_folder)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return thunk


def test(args, envs, test_episodes, n_actions, model, model_type):
    device = select_device()

    config = load_config(args.config)

    if args.model_type == "rainbow":
        V_MIN = config["v_min"]
        V_MAX = config["v_max"]
        N_ATOMS = config["n_atoms"]

    RANDOM_SEED = config["random_seed"]

    set_seed(RANDOM_SEED)

    NUM_ENVS = args.num_envs
    WARMUP_EPISODES = args.warmup_episodes

    t_observations, infos = envs.reset()
    episode_rewards = np.zeros(NUM_ENVS)
    total_rewards = []
    episode_counts = np.zeros(NUM_ENVS)

    progress_bar = tqdm(total=test_episodes, desc="Testing Progress")
    total_episodes = 0
    life_before = np.zeros(NUM_ENVS, dtype=np.int32)
    override_action = np.zeros(NUM_ENVS, dtype=np.int32)

    while total_episodes < test_episodes:
        with torch.no_grad():
            if model_type == "double":
                eps = 0.01
                if np.random.rand() < eps:
                    actions = np.random.randint(0, n_actions, size=NUM_ENVS)
                else:
                    obs_tensor = torch.tensor(t_observations, device=device)
                    q_values = model(obs_tensor)
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()
            elif model_type == "rainbow":
                q_atoms = model(torch.tensor(np.array(t_observations), device=device))
                q_values = (
                    q_atoms * torch.linspace(V_MIN, V_MAX, N_ATOMS, device=device)
                ).sum(dim=2)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Override action if life_before is different
        for i in range(NUM_ENVS):
            if override_action[i]:
                actions[i] = 1

        t1_observations, rewards, dones, _, infos = envs.step(actions)
        episode_rewards += rewards

        for i in range(NUM_ENVS):
            if dones[i]:
                total_rewards.append(episode_rewards[i])
                episode_rewards[i] = 0
                episode_counts[i] += 1
                total_episodes += 1
                if len(total_rewards) > WARMUP_EPISODES:
                    progress_bar.set_description(
                        f"Warmup Reward: {np.mean(total_rewards[:WARMUP_EPISODES]):.2f} "
                        f"Mean Reward: {np.mean(total_rewards[WARMUP_EPISODES:]):.2f}"
                    )
                progress_bar.update(1)

        # Update life_before
        for i in range(NUM_ENVS):
            if infos["lives"][i] != life_before[i]:
                override_action[i] = 1
            else:
                override_action[i] = 0
        life_before = infos["lives"]

        # Update observations
        t_observations = t1_observations

    return (
        np.mean(total_rewards[WARMUP_EPISODES:]),
        np.std(total_rewards[WARMUP_EPISODES:]),
        np.min(total_rewards[WARMUP_EPISODES:]),
        np.max(total_rewards[WARMUP_EPISODES:]),
    )


def main(args):
    config = load_config(args.config)

    device = select_device()

    GAME_NAME = config["game_name"]
    EXPERIMENT_NAME = config["experiment_name"]

    RANDOM_SEED = config["random_seed"]

    ENV_NAME = f"ALE/{GAME_NAME}-v5"
    FILE_NAME = f"{EXPERIMENT_NAME}_{GAME_NAME.lower()}_{args.model_type}"

    if args.model_type == "rainbow":
        V_MIN = config["v_min"]
        V_MAX = config["v_max"]
        N_ATOMS = config["n_atoms"]

    test_episodes = args.warmup_episodes + args.testing_episodes
    checkpoint_files = glob.glob(args.checkpoint_folder)

    checkpoint_files.sort(key=lambda x: int(x.split("checkpoint")[-1].split(".")[0]))
    print(
        f"Found {len(checkpoint_files)} checkpoint files for experiment {EXPERIMENT_NAME}"
    )

    envs = gym.vector.AsyncVectorEnv(
        [make_env(ENV_NAME, seed=RANDOM_SEED + idx) for idx in range(args.num_envs)]
    )
    n_actions = envs.single_action_space.n

    results = []
    for checkpoint_file in tqdm(checkpoint_files, desc="Testing Checkpoints"):
        print(f"Testing with checkpoint: {checkpoint_file}")
        if args.model_type == "double":
            model = DeepQNetwork(n_actions).to(device)
        elif args.model_type == "rainbow":
            model = RainbowDeepQNetwork(V_MIN, V_MAX, N_ATOMS, n_actions, device).to(
                device
            )
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")

        state_dict = torch.load(
            checkpoint_file, map_location=torch.device(device), weights_only=True
        )
        model.load_state_dict(state_dict=state_dict)
        model.eval()
        results.append(
            test(args, envs, test_episodes, n_actions, model, args.model_type)
        )

    envs.close()

    # Save results to CSV
    mean_rewards = [result[0] for result in results]
    std_rewards = [result[1] for result in results]
    min_rewards = [result[2] for result in results]
    max_rewards = [result[3] for result in results]

    df = pd.DataFrame(
        {
            "checkpoint_file": checkpoint_files,
            "mean_reward": mean_rewards,
            "std_reward": std_rewards,
            "min_reward": min_rewards,
            "max_reward": max_rewards,
        }
    )

    df.to_csv(f"test_{FILE_NAME}.csv", index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        range(len(checkpoint_files)),
        mean_rewards,
        yerr=std_rewards,
        fmt="-o",
        label="Mean Reward",
    )
    plt.fill_between(
        range(len(checkpoint_files)),
        min_rewards,
        max_rewards,
        alpha=0.2,
        label="Min/Max Reward Range",
    )
    plt.xlabel("Checkpoint Index")
    plt.ylabel("Reward")
    plt.title(f"Testing Results for {EXPERIMENT_NAME}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    args = test_parse_args()
    main(args)
