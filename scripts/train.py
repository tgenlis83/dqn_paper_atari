import numpy as np
import torch
from memory.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from memory.n_step_buffer import NStepTransitionBuffer
from networks.dqn import DeepQNetwork
from networks.rainbowdqn import RainbowDeepQNetwork
from utils.progress_bar import ProgressBar
from utils.training_log import TrainingLog
from utils.utils import make_env, select_device, set_seed, load_config, train_parse_args


def initialize_environment(config):
    GAME_NAME = config["game_name"]
    ENV_NAME = f"ALE/{GAME_NAME}-v5"
    FRAME_SKIP = config.get("frame_skip", 4)
    RANDOM_SEED = config.get("random_seed", 11)
    set_seed(RANDOM_SEED)
    env = make_env(ENV_NAME, render_mode=None, frame_skip=FRAME_SKIP)
    n_actions = env.action_space.n
    return env, n_actions


def initialize_dqn(config, n_actions, device, model_type):
    if model_type == "rainbow":
        V_MIN = config["v_min"]
        V_MAX = config["v_max"]
        N_ATOMS = config["n_atoms"]
        dqn = RainbowDeepQNetwork(V_MIN, V_MAX, N_ATOMS, n_actions, device).to(device)
        target_dqn = RainbowDeepQNetwork(V_MIN, V_MAX, N_ATOMS, n_actions, device).to(
            device
        )
    else:
        dqn = DeepQNetwork(n_actions).to(device)
        target_dqn = DeepQNetwork(n_actions).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    optimizer = torch.optim.Adam(dqn.parameters(), lr=config["learning_rate"])
    return dqn, target_dqn, optimizer


def initialize_buffers(config, model_type):
    REPLAY_MEMORY_SIZE = config["replay_memory_size"]
    MINI_BATCH_SIZE = config["mini_batch_size"]
    ALPHA = config.get("alpha", 0.6)
    if model_type == "rainbow":
        buffer = PrioritizedReplayBuffer(
            REPLAY_MEMORY_SIZE, alpha=ALPHA, batch_size=MINI_BATCH_SIZE
        )
        n_step_buffer = NStepTransitionBuffer(
            config["n_step"], config["discount_factor"]
        )
    else:
        buffer = ReplayBuffer(REPLAY_MEMORY_SIZE)
        n_step_buffer = None
    return buffer, n_step_buffer


def update_target_network(t, dqn, target_dqn, TARGET_UPDATE_FREQ):
    if t % TARGET_UPDATE_FREQ == 0:
        target_dqn.load_state_dict(dqn.state_dict())


def save_checkpoint(t, dqn, file_name, SAVE_FREQUENCY, training_log):
    if t > 0 and t % SAVE_FREQUENCY == 0:
        torch.save(dqn.state_dict(), f"{file_name}_checkpoint{t:08d}.pt")
        training_log.save()


def train(config, device, model_type):
    env, n_actions = initialize_environment(config)
    dqn, target_dqn, optimizer = initialize_dqn(config, n_actions, device, model_type)
    buffer, n_step_buffer = initialize_buffers(config, model_type)
    training_log = TrainingLog(config["experiment_name"])
    progress_bar = ProgressBar(config["max_steps"])

    t_observation, info = env.reset()
    t_observation = np.array(t_observation, copy=False)
    life_before = 0
    override_action = False

    for t in progress_bar:
        # Exploration strategy
        if model_type == "rainbow":
            dqn.reset_noise()
            with torch.no_grad():
                obs_tensor = torch.tensor(
                    np.array(t_observation, copy=False),
                    device=device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                q_atoms = dqn(obs_tensor)
                q_values = (q_atoms * dqn.support).sum(dim=2)
                action = q_values.argmax(dim=1).item()
        else:
            eps = max(
                config["min_epsilon"],
                config["min_epsilon"]
                + (config["max_epsilon"] - config["min_epsilon"])
                * (1 - t / (config["epsilon_phase"] * config["max_steps"])),
            )
            if np.random.rand(1) < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_tensor = torch.tensor(
                        np.array(t_observation, copy=False), device=device
                    ).unsqueeze(0)
                    q_values = dqn(obs_tensor)
                    action = torch.argmax(q_values, dim=1).item()

        if override_action:
            action = 1
            override_action = False

        # Environment interaction
        t1_observation, reward, done, truncated, info = env.step(action)
        t1_observation = np.array(t1_observation, copy=False)
        if info["lives"] != life_before:
            override_action = True
        life_before = info["lives"]

        # Log reward
        training_log.episode_reward += reward
        reward = (
            np.clip(reward, -1, 1)
            if model_type == "rainbow"
            else np.sign(reward)
        )

        # Store transitions
        transition = (t_observation, action, reward, t1_observation, done)
        if n_step_buffer:
            n_step_transition = n_step_buffer.store(transition)
            if n_step_transition:
                buffer.store(*n_step_transition)
        else:
            buffer.append(t_observation, t1_observation, action, reward, done)

        if done or truncated:
            while n_step_buffer and len(n_step_buffer.buffer) > 0:
                n_step_transition = n_step_buffer._get_n_step_transition()
                buffer.store(*n_step_transition)
                n_step_buffer.buffer.popleft()
            n_step_buffer and n_step_buffer.reset()
            if info["lives"] == 0:
                training_log.log_episode(t)
                progress_bar.update(training_log, buffer)
                training_log.reset_episode()
            t_observation, info = env.reset()
            t_observation = np.array(t_observation, copy=False)
            life_before = info["lives"]
        else:
            t_observation = t1_observation
            training_log.episode_steps += 1

        # Training
        if t > config["replay_start_size"] and t % 4 == 0:
            beta = (
                min(
                    1.0,
                    config["beta_start"]
                    + (t - config["replay_start_size"])
                    * (1.0 - config.get("BETA_START", 0.4))
                    / (config["max_steps"] - config["replay_start_size"]),
                )
                if model_type == "rainbow"
                else None
            )

            if model_type == "rainbow":
                (
                    states,
                    next_states,
                    actions_batch,
                    rewards_batch,
                    dones_batch,
                    is_weights,
                    indices,
                ) = buffer.sample_batch(beta, device)
                dqn.reset_noise()
                with torch.no_grad():
                    next_q_atoms_online = dqn(next_states)
                    next_q_values = (next_q_atoms_online * dqn.support).sum(dim=2)
                    next_actions = next_q_values.argmax(dim=1)
                    next_q_atoms_target = target_dqn(next_states)
                    next_dist = next_q_atoms_target[
                        range(config["mini_batch_size"]), next_actions
                    ]
                    target_dist = target_dqn.project_distribution(
                        next_dist,
                        rewards_batch,
                        dones_batch,
                        config["discount_factor"] ** config["n_step"],
                    )
                q_atoms = dqn(states)
                current_dist = q_atoms[
                    range(config["mini_batch_size"]), actions_batch.long()
                ]
                loss = -torch.sum(target_dist * torch.log(current_dist + 1e-8), dim=1)
                prios = loss + 1e-6
                loss = (loss * is_weights).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                indices = indices.astype(np.int64)
                buffer.update_priorities(indices, prios.detach().cpu().numpy())
                training_log.episode_loss += loss.item()
            else:
                obs_batch, next_obs_batch, actions, rewards, dones = (
                    buffer.get_minibatch(config["mini_batch_size"], device=device)
                )
                with torch.no_grad():
                    not_done = ~dones.bool()
                    a_prime = target_dqn(next_obs_batch).amax(dim=1)
                    y_j = rewards + config["discount_factor"] * a_prime * not_done
                optimizer.zero_grad()
                q_values = dqn(obs_batch)
                idx = torch.arange(actions.size(0)).to(device).long()
                values = q_values[idx, actions.squeeze().long()]
                loss = torch.nn.functional.huber_loss(y_j, values)
                loss.backward()
                optimizer.step()
                training_log.episode_loss += loss.item()

        update_target_network(t, dqn, target_dqn, config["target_update_freq"])
        save_checkpoint(
            t, dqn, config["experiment_name"], config["save_frequency"], training_log
        )

    env.close()


if __name__ == "__main__":
    args = train_parse_args()
    config = load_config(args.config)
    train(config, select_device(), args.model_type)
