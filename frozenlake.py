import gym

# Create the FrozenLake environment (deterministic version for initial setup)
env = gym.make('FrozenLake-v1', is_slippery=False)

# Reset the environment to get the initial state
state = env.reset()

print("Initial state:", state)
print("Environment render:")
env.render()
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)