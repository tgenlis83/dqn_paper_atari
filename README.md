# Deep Q-Network (DQN) Implementation and Upgrades

*Make sure to read the report from the repository for more details [report.pdf](report.pdf).*

## Introduction

This project presents the re-implementation and enhancement of the Deep Q-Network (DQN) originally introduced by Mnih et al. in 2015. Motivated by a desire to deepen our understanding of reinforcement learning (RL) and deep learning, and driven by curiosity about reimplementing influential papers, we replicated the DQN with modifications to accommodate computational constraints.

We then upgraded the model by integrating components from the Rainbow algorithm, including dueling networks, prioritized experience replay, n-step returns, and noisy networks. Comprehensive evaluations were conducted by comparing each version against a random action baseline and analyzing the impact of different parameters and network configurations.

The results demonstrate significant performance improvements with each enhancement, highlighting the effectiveness of the integrated components. This work provides insights into the practical implementation of advanced RL techniques and underscores the educational value of reimplementing foundational research.

## Motivation

Our primary motivation for this project was to deepen our understanding of reinforcement learning and deep learning by engaging directly with seminal research in the field. By reimplementing the original DQN, we aimed to explore its foundational mechanics firsthand. Enhancing the DQN with components from the Rainbow algorithm allowed us to evaluate the effects of these upgrades on performance and further our educational growth in advanced RL techniques.

### Environment

- **Game**: Atari 2600 Breakout
- **Reason**: Manageable complexity and availability of comparative metrics.

### Baseline Comparison

To ensure that improvements are meaningful and not due to environmental dynamics, we compared the agent's performance to a random action baseline.

## Upgrades Using Rainbow Components

We enhanced the original DQN implementation by integrating several components from the Rainbow algorithm:

1. **Dueling Network Architecture**: Separates the estimation of state value and advantage for each action.
2. **Prioritized Experience Replay**: Samples important transitions more frequently based on temporal-difference (TD) error.
3. **N-Step Returns**: Uses multi-step returns for better learning from delayed rewards.
4. **Noisy Networks**: Adds stochasticity to network weights to improve exploration.

## Results

### Performance Metrics

The enhanced DQN showed improved performance over the classic DQN and significantly outperformed the random action baseline. The results demonstrate the effectiveness of integrating advanced components from the Rainbow algorithm.

## Repository Structure

- [`/notebooks/`](notebooks/): Jupyter notebooks containing the code for each version of the implementation.
- [`/checkpoints/`](checkpoints/): Saved model checkpoints at different timesteps during training.
- [`/data/`](data/): Plot data and logs generated during training.
- [`/report.pdf`](report.pdf): The full report detailing the implementation and results.

## References

- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- Hessel, M., et al. (2018). *Rainbow: Combining improvements in deep reinforcement learning*. AAAI Conference.

## Acknowledgments

We would like to thank the authors of the original papers for their foundational work in reinforcement learning and deep learning.