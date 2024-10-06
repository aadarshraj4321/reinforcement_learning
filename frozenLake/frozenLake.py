import gym
import numpy as np
import time
import pickle as pkl



env = gym.make("FrozenLake-v1", is_slippery=True)
qTable = np.zeros(shape=(16, 4))

# print(qTable)


def policy(state, explore=0.0):
    if(np.random.random() <= explore):
        return np.random.randint(0, 4)
    else:
        return int(np.argmax(qTable[state]))


## Parameters
EPSILON = 1.0
EPSILON_DECAY = 1.001
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODE = 50000




for episode in range(NUM_EPISODE):
    totalEpisode = 0
    totalReward = 0

    done = False
    state, _ = env.reset()
    while not done:
        action = policy(state, EPSILON)
        nextState, reward, done, truncated, info = env.step(action)
        nextAction = policy(nextState)

        qTable[state][action] += ALPHA * (reward + GAMMA * qTable[nextState][nextAction] - qTable[state][action])
        state = nextState

        totalReward += reward
        totalEpisode += 1

        done = done or truncated

        #env.render()

    print("EPISODE: {} || TOTAL_EPISODE: {} || TOTAL_REWARD: {} || EPSILON: {}".format(episode, totalEpisode, totalReward, EPSILON))
    EPSILON /= EPSILON_DECAY


env.close()
pkl.dump(qTable, open("frozenLakeQLearningModel.pkl", "wb"))