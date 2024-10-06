import numpy as np
import gym

# Create the CarRacing environment
env = gym.make("CarRacing-v2", render_mode="human")

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.995  # Decay rate of epsilon after each episode
min_epsilon = 0.01
num_episodes = 1000

# Discretization of state and action spaces
state_space_size = (40, 40, 40)  # Example state space discretization (position, velocity, etc.)
action_space_size = (3, 3, 3)  # 3 discrete values each for steering, acceleration, and braking

# Initialize the Q-table with zeros, reshape to handle state-action pairs as a flat array
qTable = np.zeros(state_space_size + action_space_size)

# Discretize the state space (example method)
def discretize_state(state):
    """ Convert continuous state to discrete state """
    state_low = env.observation_space.low
    state_high = env.observation_space.high
    normalized_state = (state - state_low) / (state_high - state_low)
    discrete_state = np.floor(normalized_state * np.array(state_space_size)).astype(int)
    return tuple(np.clip(discrete_state, 0, np.array(state_space_size) - 1))

# Discretize the action space (3 discrete actions: steering, acceleration, brake)
def discretize_action(action):
    """ Convert continuous action to discrete """
    steering = np.digitize(action[0], np.linspace(-1, 1, 3)) - 1  # Steering: -1, 0, 1
    acceleration = np.digitize(action[1], np.linspace(0, 1, 3)) - 1  # Acceleration: 0, 0.5, 1
    brake = np.digitize(action[2], np.linspace(0, 1, 3)) - 1  # Brake: 0, 0.5, 1
    return (steering, acceleration, brake)

# Reverse discretized actions back to continuous space for environment step
def reverse_discretize_action(action):
    """ Convert discrete action back to continuous """
    steering, acceleration, brake = action
    steering = np.linspace(-1, 1, 3)[steering]
    acceleration = np.linspace(0, 1, 3)[acceleration]
    brake = np.linspace(0, 1, 3)[brake]
    return np.array([steering, acceleration, brake])

# Epsilon-greedy policy
def policy(state, epsilon):
    """ Choose action using epsilon-greedy policy """
    if np.random.random() < epsilon:
        # Explore: take random action
        return tuple(np.random.randint(0, size) for size in action_space_size)
    else:
        # Exploit: take action with max Q-value
        return np.unravel_index(np.argmax(qTable[state]), action_space_size)

# Main SARSA loop
for episode in range(num_episodes):
    done = False
    total_reward = 0
    
    # Reset the environment
    state, _ = env.reset()
    state = discretize_state(state)
    
    # Choose initial action
    action = policy(state, epsilon)
    
    while not done:
        # Convert the discrete action back to continuous for the environment
        continuous_action = reverse_discretize_action(action)
        
        # Take a step in the environment
        next_state, reward, done, truncated, info = env.step(continuous_action)
        next_state = discretize_state(next_state)
        
        # Choose the next action
        next_action = policy(next_state, epsilon)
        
        # SARSA update
        qTable[state][action] += alpha * (reward + gamma * qTable[next_state][next_action] - qTable[state][action])
        
        # Update state and action
        state = next_state
        action = next_action
        
        # Accumulate reward
        total_reward += reward
        
        # Render the environment
        env.render()
    
    # Decay epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)
    
    # Print episode information
    print(f"Episode: {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

# Close the environment
env.close()

# Save the Q-table
np.save("sarsa_qTable_carracing.npy", qTable)
