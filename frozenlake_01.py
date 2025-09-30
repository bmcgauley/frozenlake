import gymnasium as gym
import time 
# Create the FrozenLake environment (deterministic version for initial setup)
# env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='ansi')
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode='ansi') 

# Reset the environment to get the initial state
state, info = env.reset() 
env.render()
# state = env.reset()

print("Initial state:", state)
print("Environment render:")
print(env.render())
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)



done = False 
state, info = env.reset() 
env.render() 
 
while not done: 
    action = env.action_space.sample()  # Pick a random action 
    state, reward, done, truncated, info = env.step(action) 
    env.render() 
    time.sleep(1)  # Pause to see each move 
    if done: 
        print(f"Episode finished. Reward: {reward}") 
        break