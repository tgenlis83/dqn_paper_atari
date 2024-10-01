import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode="valid")


def plot_dict(plot_infos, REPLAY_START_SIZE):
    # Define the window size for moving average
    window_size = 500

    # Calculate the moving averages
    smoothed_rewards = moving_average(plot_infos["total_reward"], window_size)
    smoothed_q_values = moving_average(plot_infos["total_q_values"], window_size)
    smoothed_loss = moving_average(plot_infos["total_loss"], window_size)

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot smoothed total rewards
    ax1.plot(
        plot_infos["total_steps"][: len(smoothed_rewards)],
        smoothed_rewards,
        label="Total Reward (Smoothed)",
        color="green",
    )
    ax1.set_ylabel("Total Reward (Smoothed)", color="green")
    ax1.tick_params(axis="y", labelcolor="green")
    ax1.grid(True)  # Add grid to ax1

    # Add vertical line at REPLAY_START_SIZE
    ax1.axvline(REPLAY_START_SIZE, color='gray', linestyle='--', label='Replay Start Size')

    # Create another secondary y-axis for smoothed Q-values
    ax2 = ax1.twinx()
    ax2.spines["right"].set_position(("outward", 60))
    ax2.plot(
        plot_infos["total_steps"][: len(smoothed_q_values)],
        smoothed_q_values,
        label="Q-Value (Smoothed)",
        color="blue",
    )
    ax2.set_ylabel("Q-Value (Smoothed)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Create another secondary y-axis for smoothed loss
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 120))
    ax3.plot(
        plot_infos["total_steps"][: len(smoothed_loss)],
        smoothed_loss,
        label="Loss (Smoothed)",
        color="red",
    )
    ax3.set_ylabel("Loss (Smoothed)", color="red")
    ax3.tick_params(axis="y", labelcolor="red")

    # Combine legends from all axes
    lines, labels = [], []

    for ax in [ax1, ax2, ax3]:
        line, label = ax.get_legend_handles_labels()
        lines.extend(line)
        labels.extend(label)

    ax1.legend(lines, labels, loc="upper left")

    fig.tight_layout()
    plt.title("Training Progress")
    plt.show()