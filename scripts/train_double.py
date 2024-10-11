import numpy as np
import torch
from memory.replay_buffer import ReplayBuffer
from networks.dqn import DeepQNetwork
from utils.progress_bar import ProgressBar
from utils.training_log import TrainingLog
from utils.utils import make_env, select_device, set_seed

device = select_device()

GAME_NAME = "Assault"
ENV_NAME = f"ALE/{GAME_NAME}-v5"
EXPERIMENT_NAME = f"{GAME_NAME.lower()}_double"

LEARNING_RATE = 0.00025
DISCOUNT_FACTOR = 0.99
REPLAY_MEMORY_SIZE = 300_000
MINI_BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 32_000
FRAME_SKIP = 4
MIN_EPSILON = 0.1
MAX_EPSILON = 1.0
EPSILON_PHASE = 0.1
MAX_STEPS = 1_500_001
REPLAY_START_SIZE = 80_000
SAVE_FREQUENCY = 500_000

RANDOM_SEED = 42
set_seed(RANDOM_SEED)

def train():
    env = make_env(ENV_NAME)

    n_actions = env.action_space.n
    dqn = DeepQNetwork(n_actions).to(device)
    optimizer = torch.optim.Adam(dqn.parameters(), lr=LEARNING_RATE)

    buffer = ReplayBuffer(REPLAY_MEMORY_SIZE, (4, 84, 84), (1,))
    training_log = TrainingLog(EXPERIMENT_NAME)

    t_observation, _ = env.reset()

    progress_bar = ProgressBar(MAX_STEPS)

    for t in progress_bar:
        eps = max(
            MIN_EPSILON,
            MIN_EPSILON
            + (MAX_EPSILON - MIN_EPSILON) * (1 - t / (EPSILON_PHASE * MAX_STEPS)),
        )

        if np.random.rand(1) < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = dqn(
                    torch.tensor(np.array(t_observation), device=device).unsqueeze(0)
                )
                action = torch.argmax(q_values, dim=1).item()
                training_log.episode_q_values += q_values.mean().item()

        t1_observation, reward, done, _, info = env.step(action)
        buffer.append(
            t_observation, t1_observation, np.array([action]), np.sign(reward), done
        )
        training_log.episode_reward += reward

        if "final_info" in info:
            training_log.log_episode(t)
            progress_bar.update(training_log, buffer)
            training_log.reset_episode()
            t_observation, _ = env.reset()
        else:
            t_observation = t1_observation
            training_log.episode_steps += 1

        if t > 0 and t % SAVE_FREQUENCY == 0:
            torch.save(dqn.state_dict(), f"checkpoint{t}.pt")
            training_log.save()

        if t > REPLAY_START_SIZE:
            if t % 4 == 0:
                t_obs, t1_obs, actions, rewards, dones = buffer.get_minibatch(
                    MINI_BATCH_SIZE, device=device
                )

                with torch.no_grad():
                    not_done = ~dones.bool()
                    a_prime = dqn(t1_obs).amax(dim=1)
                    y_j = rewards + DISCOUNT_FACTOR * a_prime * not_done

                optimizer.zero_grad()

                q_values = dqn(t_obs)
                idx = torch.arange(actions.size(0)).to(device).long()
                values = q_values[idx, actions.squeeze().long()]

                loss = torch.nn.functional.huber_loss(y_j, values)

                loss.backward()
                optimizer.step()
                training_log.episode_loss += loss.item()
            if t % TARGET_UPDATE_FREQ == 0:
                dqn.target.load_state_dict(dqn.state_dict())

    env.close()

if __name__ == "__main__":
    train()