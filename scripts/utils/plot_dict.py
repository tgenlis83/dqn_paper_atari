import numpy as np
import matplotlib.pyplot as plt

def plot_dict(title, plot_infos, REPLAY_START_SIZE, dpi=200):
    smoothed_rewards = plot_infos["episode_rewards_ma"]
    smoothed_rewards_min = plot_infos["episode_rewards_min"]
    smoothed_rewards_max = plot_infos["episode_rewards_max"]
    
    smoothed_q_values = plot_infos["mean_q_value_ma"]
    smoothed_loss = plot_infos["loss_ma"]

    fig, ax1 = plt.subplots(figsize=(12, 5), dpi=dpi)

    # Plot smoothed total rewards
    ax1.plot(
        plot_infos["steps"][: len(smoothed_rewards)],
        smoothed_rewards,
        label="Total Reward",
        color="C2",
    )
    ax1.set_ylabel("Total Reward", color="C2")
    ax1.tick_params(axis="y", labelcolor="C2")

    # Add vertical line at REPLAY_START_SIZE
    ax1.axvline(REPLAY_START_SIZE, color='gray', linestyle='--', label='Replay Start Size')

    # Fill between min and max of smoothed rewards
    ax1.fill_between(
        plot_infos["steps"][: len(smoothed_rewards)],
        smoothed_rewards_min,
        smoothed_rewards_max,
        color="C2",
        alpha=0.1,
        label="Reward Range"
    )
    ax1.grid(True)  # Add grid to ax1

    # Create another secondary y-axis for smoothed Q-values
    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("outward", 60))
    ax2.plot(
        plot_infos["steps"][: len(smoothed_q_values)],
        smoothed_q_values,
        label="Mean Q-Value",
        color="C0",
    )
    ax2.set_ylabel("Mean Q-Value", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")

    # Create another secondary y-axis for smoothed loss
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 120))
    ax3.plot(
        plot_infos["steps"][: len(smoothed_loss)],
        smoothed_loss,
        label="Loss",
        color="C1",
    )
    ax3.set_ylabel("Loss", color="C1")
    ax3.tick_params(axis="y", labelcolor="C1")

    # Combine legends from all axes
    lines, labels = [], []

    for ax in [ax1, ax2, ax3]:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)

    ax3.legend(lines, labels, loc="upper left", borderaxespad=1.).set_zorder(2)

    fig.tight_layout()
    plt.title(title)
    plt.show()