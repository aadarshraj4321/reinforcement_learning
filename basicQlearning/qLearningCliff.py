import gym
import pickle as pkl
import numpy as np
import time


env = gym.make("CliffWalking-v0")
qTable = np.zeros(shape=(48, 4))


# Parameters
EPSILION = 0.1
GAMMA = 0.9
ALPHA = 0.1
NUM_EPISODES = 500

def policy(state, explore=0.0):
    if(np.random.random() <= explore):
        return np.random.randint(0, 4)
    else:
        return int(np.argmax(qTable[state]))


for episode in range(NUM_EPISODES):

    totalReward = 0
    totalEpisode = 0

    done = False
    state, _ = env.reset()

    while not done:
        action = policy(state, EPSILION)
        nextState, reward, done, truncated, info = env.step(action)

        nextAction = policy(nextState)
        qTable[state][action] += ALPHA * (reward + GAMMA * qTable[nextState][nextAction] - qTable[state][action])

        state = nextState

        totalReward += reward
        totalEpisode += 1
    
    print("EPISODE: {} || TOTAL_EPISODE: {} || TOTAL_REWARD: {}".format(episode, totalEpisode, totalReward))

env.close()
pkl.dump(qTable, open("qLearningQTable.pkl", "wb"))
