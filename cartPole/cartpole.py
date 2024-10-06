import gym
import numpy as np
import tensorflow as tf



env = gym.make("CartPole-v1", render_mode="human")


for episode in range(5):
    done = False
    state = env.reset()

    while not done:
        action = tf.random.uniform(shape=(), minval=0, maxval=2, dtype=tf.int32).numpy()
        print("Action : {}".format(action))
        nextState, reward, done, truncated, info = env.step(action)
        env.render()
    
env.close()
