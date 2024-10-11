import gymnasium as gym
import torch
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import AtariPreprocessing, FrameStack, RecordEpisodeStatistics, AutoResetWrapper
import glob
import pandas as pd
import matplotlib.pyplot as plt

from networks.dqn import DeepQNetwork
from utils.utils import select_device

device = select_device()

RANDOM_SEED = 42
NUM_ENVS = 32

def make_env(env_id, seed=None):
    def thunk():
        env = gym.make(env_id, frameskip=1, max_episode_steps=108_000)
        env = AtariPreprocessing(env, frame_skip=4, terminal_on_life_loss=False)
        env = FrameStack(env, 4)
        env = RecordEpisodeStatistics(env)
        env = AutoResetWrapper(env)
        if seed is not None:
            env.reset(seed=seed)
        return env
    return thunk

def test(envs, n_actions, path):
    dqn = DeepQNetwork(n_actions).to(device)
    state_dict = torch.load(
        path,
        map_location=torch.device(device),
        weights_only=True
    )
    dqn.load_state_dict(state_dict=state_dict)
    dqn.eval()

    t_observations, infos = envs.reset()
    episode_rewards = np.zeros(NUM_ENVS)
    total_rewards = []
    episode_counts = np.zeros(NUM_ENVS)

    progress_bar = tqdm(total=TEST_EPISODES, desc="Testing Progress")
    total_episodes = 0

    while total_episodes < TEST_EPISODES:
        with torch.no_grad():
            eps = 0.01
            if np.random.rand() < eps:
                actions = np.random.randint(0, n_actions, size=NUM_ENVS)
            else:
                obs_tensor = torch.tensor(t_observations, device=device)
                q_values = dqn(obs_tensor)
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        t1_observations, rewards, dones, truncateds, infos = envs.step(actions)
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

        # Update observations
        t_observations = t1_observations
    
    return (
        np.mean(total_rewards[WARMUP_EPISODES:]),
        np.std(total_rewards[WARMUP_EPISODES:]),
        np.min(total_rewards[WARMUP_EPISODES:]),
        np.max(total_rewards[WARMUP_EPISODES:])
    )

if __name__ == "__main__":
    WARMUP_EPISODES = 100  # Define the number of episodes to warm up
    TEST_EPISODES = WARMUP_EPISODES + 500  # Define the number of episodes to test
    # Find all checkpoint files
    checkpoint_files = glob.glob("../09.10.24/a=>b ddqn/a=>b_double_checkpoint*.pt")
    # Sort checkpoint files by their name human style
    checkpoint_files.sort(key=lambda x: int(x.split('checkpoint')[-1].split('.')[0]))
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    envs = gym.vector.AsyncVectorEnv([make_env("ALE/Breakout-v5", seed=RANDOM_SEED + idx) for idx in range(NUM_ENVS)])
    n_actions = envs.single_action_space.n
    
    results = []
    for checkpoint_file in tqdm(checkpoint_files, desc="Testing Checkpoints"):
        print(f"Testing with checkpoint: {checkpoint_file}")
        results.append(test(envs, n_actions, checkpoint_file))
    
    envs.close()
    
    mean_rewards = [result[0] for result in results]
    std_rewards = [result[1] for result in results]
    min_rewards = [result[2] for result in results]
    max_rewards = [result[3] for result in results]
    
    # Create a DataFrame from the results
    df = pd.DataFrame({
        'checkpoint_file': checkpoint_files,
        'mean_reward': mean_rewards,
        'std_reward': std_rewards,
        'min_reward': min_rewards,
        'max_reward': max_rewards
    })

    # Save the DataFrame to a CSV file
    df.to_csv('test_a=>b_dqn_breakout_results.csv', index=False)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.errorbar(range(len(checkpoint_files)), mean_rewards, yerr=std_rewards, fmt='-o', label='Mean Reward')
    plt.fill_between(range(len(checkpoint_files)), min_rewards, max_rewards, alpha=0.2, label='Min/Max Reward Range')
    plt.xlabel('Checkpoint Index')
    plt.ylabel('Reward')
    plt.title('Performance of DQN on Breakout-v5')
    plt.legend()
    plt.grid(True)
    plt.show()
