import gymnasium as gym

# Create the Breakout environment with rendering enabled
env = gym.make('ALE/Breakout-v5', render_mode='human')

# Reset the environment to get the initial observation and info
observation, info = env.reset()

for _ in range(10000):  # Run for 10,000 steps
    # Sample a random action from the action space
    action = env.action_space.sample()
    
    # Take the action and receive the new state and other info
    observation, reward, terminated, truncated, info = env.step(action)
    
    # If the episode is over, reset the environment
    if terminated or truncated:
        observation, info = env.reset()

# Close the environment when done
env.close()