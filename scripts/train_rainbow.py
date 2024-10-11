import numpy as np
import torch
from memory.replay_buffer import PrioritizedReplayBuffer
from networks.rainbowdqn import RainbowDeepQNetwork
from utils.progress_bar import ProgressBar
from utils.training_log import TrainingLog
from utils.utils import make_env, select_device, set_seed

device = select_device()

GAME_NAME = "Assault"
ENV_NAME = f"ALE/{GAME_NAME}-v5"
EXPERIMENT_NAME = f"{GAME_NAME.lower()}_rainbow"

LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
REPLAY_MEMORY_SIZE = 300_000
MINI_BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 32_000
FRAME_SKIP = 4
MAX_STEPS = 5_000_001
REPLAY_START_SIZE = 80_000
SAVE_FREQUENCY = 500_000

V_MIN = -10
V_MAX = 10
N_ATOMS = 51
DELTA_Z = (V_MAX - V_MIN) / (N_ATOMS - 1)

N_STEP = 3
ALPHA = 0.5
BETA_START = 0.4
BETA_FRAMES = MAX_STEPS - REPLAY_START_SIZE

RANDOM_SEED = 27
set_seed(RANDOM_SEED)


def train():
    env = make_env(ENV_NAME)
    n_actions = env.action_space.n
    dqn = RainbowDeepQNetwork(V_MIN, V_MAX, N_ATOMS, n_actions).to(device)
    dqn_prime = RainbowDeepQNetwork(V_MIN, V_MAX, N_ATOMS, n_actions).to(device)
    dqn_prime.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    buffer = PrioritizedReplayBuffer(
        REPLAY_MEMORY_SIZE,
        alpha=ALPHA,
        obs_shape=(4, 84, 84),
        batch_size=MINI_BATCH_SIZE,
        n_step=N_STEP,
        gamma=DISCOUNT_FACTOR,
    )

    training_log = TrainingLog(EXPERIMENT_NAME)

    t_observation, _ = env.reset(seed=RANDOM_SEED)

    progress_bar = ProgressBar(MAX_STEPS)

    for t in progress_bar:
        dqn.reset_noise()
        with torch.no_grad():
            obs_tensor = torch.tensor(
                np.array(t_observation), device=device, dtype=torch.uint8
            ).unsqueeze(0)
            q_atoms = dqn(obs_tensor)
            q_values = (q_atoms * dqn.support).sum(dim=2)
            action = q_values.argmax(dim=1).item()
            training_log.episode_q_values += q_values.mean().item()

        t1_observation, reward, done, _, info = env.step(action)
        training_log.episode_reward += reward
        reward = np.clip(reward, -1, 1)

        buffer.store(t_observation, action, reward, t1_observation, done)

        if done:
            training_log.log_episode(t)
            progress_bar.update(training_log, buffer)
            training_log.reset_episode()
            t_observation, _ = env.reset()
        else:
            t_observation = t1_observation
            training_log.episode_steps += 1

        if t > REPLAY_START_SIZE:
            if t % 4:
                beta = min(
                    1.0,
                    BETA_START
                    + (t - REPLAY_START_SIZE) * (1.0 - BETA_START) / BETA_FRAMES,
                )

                (
                    states,
                    actions_batch,
                    rewards_batch,
                    next_states,
                    dones_batch,
                    is_weights,
                    indices,
                ) = buffer.sample_batch(beta, device)

                dqn.reset_noise()
                dqn_prime.reset_noise()

                with torch.no_grad():
                    next_q_atoms_online = dqn(next_states)
                    next_q_values = (next_q_atoms_online * dqn.support).sum(dim=2)
                    next_actions = next_q_values.argmax(dim=1)

                    next_q_atoms_target = dqn_prime(next_states)
                    next_dist = next_q_atoms_target[
                        range(MINI_BATCH_SIZE), next_actions
                    ]
                    target_dist = dqn_prime.project_distribution(
                        next_dist,
                        rewards_batch,
                        dones_batch,
                        DISCOUNT_FACTOR**N_STEP,
                        device,
                    )

                q_atoms = dqn(states)
                current_dist = q_atoms[range(MINI_BATCH_SIZE), actions_batch]

                loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1)
                prios = loss + 1e-6
                loss = (loss * is_weights).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                buffer.update_priorities(indices, prios.detach().cpu().numpy())
                training_log.episode_loss += loss.item()

            if t % TARGET_UPDATE_FREQ == 0:
                dqn_prime.load_state_dict(dqn.state_dict())

        if t > 0 and t % SAVE_FREQUENCY == 0:
            torch.save(dqn.state_dict(), f"{EXPERIMENT_NAME}_checkpoint{t:08d}.pt")
            training_log.save()

    env.close()


if __name__ == "__main__":
    train()
