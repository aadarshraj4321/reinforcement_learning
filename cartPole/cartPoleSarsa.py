import gym
import tensorflow as tf
from keras import Model, Input
from keras.layers import Dense

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Q Network 
network_input = Input(shape=(4,))  # State dimension is 4 for CartPole
x = Dense(64, activation="relu")(network_input)
x = Dense(32, activation="relu")(x)
output = Dense(2, activation="linear")(x)  # 2 actions: left or right
qNetwork = Model(inputs=network_input, outputs=output)

# Parameters
ALPHA = 0.001
EPSILON = 1.0
EPSILON_DECAY = 1.001
GAMMA = 0.99
NUM_EPISODES = 500

# Policy function to decide actions
def policy(state, explore=0.0):
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        return tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    else:
        return tf.argmax(qNetwork(state)[0], output_type=tf.int32)

# Main training loop
for episode in range(NUM_EPISODES):

    totalReward = 0
    totalEpisode = 0
    done = False

    # Reset environment and extract only the initial state
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)  # Convert state to tensor
    action = policy(state, EPSILON)

    while not done:
        # Take the action and observe next state, reward, and done
        nextState, reward, done, truncated, info = env.step(action.numpy())
        nextState = tf.convert_to_tensor([nextState], dtype=tf.float32)  # Convert next state to tensor
        nextAction = policy(nextState, EPSILON)

        # Compute the target using the Bellman equation
        target = reward + GAMMA * qNetwork(nextState)[0][nextAction]
        if done:
            target = reward  # If done, the target is just the reward

        # Compute gradients and perform weight updates
        with tf.GradientTape() as tape:
            current = qNetwork(state)
        gradient = tape.gradient(current, qNetwork.trainable_weights)

        delta = target - current[0][action]  # Temporal difference error

        for j in range(len(gradient)):
            qNetwork.trainable_weights[j].assign_add(ALPHA * delta * gradient[j])

        # Move to the next state
        state = nextState
        action = nextAction

        totalReward += reward
        totalEpisode += 1

        #env.render()

    
    print(f"EPISODE: {episode} || TOTAL_EPISODE: {totalEpisode} || TOTAL_REWARD: {totalReward} || EPSILON: {EPSILON}")
    EPSILON /= EPSILON_DECAY

qNetwork.save("sarsaModelQNetwork.keras")
env.close()
