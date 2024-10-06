import tensorflow as tf
import gym
import numpy as np
from keras.models import load_model

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Load the trained Q-Network
qNetwork = load_model("sarsaModelQNetwork.keras")

# Policy function to select actions
def policy(state, explore=0.0):
    if tf.random.uniform(shape=(), maxval=1) <= explore:
        return tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32)
    else:
        return tf.argmax(qNetwork(state)[0], output_type=tf.int32)

# Run episodes
for episode in range(5):
    done = False
    state, _ = env.reset()
    state = tf.convert_to_tensor([state], dtype=tf.float32)  # Convert state to tensor

    while not done:
        # Select the action using the policy
        action = policy(state).numpy()  # Convert action to NumPy integer
        # Take the action in the environment
        state, reward, done, truncated, info = env.step(action)
        state = tf.convert_to_tensor([state], dtype=tf.float32)  # Convert next state to tensor

        env.render()

env.close()
