import gym
import numpy as np
import time 
import pickle as pkl


env = gym.make("CliffWalking-v0", render_mode="human")

qTable = pkl.load(open("basicQlearning\qLearningQTable.pkl", "rb"))


def policy(state, explore=0.0):
    if np.random.random() <= explore:
        return np.random.randint(0, 4)
    else:
        return int(np.argmax(qTable[state]))


NUM_EPISODE = 5

for episode in range(NUM_EPISODE):
    totalReward = 0
    totalEpisode = 0
    done = False
    state, _ = env.reset()

    while not done:
        action = policy(state)
        state, reward, done, truncated, info = env.step(action)
        totalReward += reward
        totalEpisode += 1
        env.render()
    
    print("EPISODE: {} || TOTAL EPISODE: {} || TOTAL REWARD: {}".format(episode, totalEpisode, totalReward))

env.close()

