import gym
import time
import numpy as np
import pickle as pkl

env = gym.make("CliffWalking-v0")

qTable = np.zeros(shape=(48, 4))
#print(qTable)

def policy(state, explore=0.1):
    if np.random.random() <= explore:
        return np.random.randint(0, 4)
    else:
        return int(np.argmax(qTable[state]))

## Parameters
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
NUM_EPISODES = 500


for episode in range(NUM_EPISODES):
    done = False
    totalReward = 0
    episodeLength = 0

    state, _ = env.reset()
    action = policy(state, EPSILON)

    while not done:
        nextState, reward, done, truncated, info = env.step(action)
        nextAction = policy(nextState, EPSILON)

        qTable[state][action] += ALPHA * (reward + GAMMA * qTable[nextState][nextAction] - qTable[state][action])
        state = nextState
        action = nextAction

        totalReward += reward
        episodeLength += 1



        done = done or truncated
        print("Episode: {} || Episode Length: {} || Total Reward: {}".format(episode, episodeLength, totalReward))

    
    # if(episode % 100 == 0):
    #     print("Episode: {} || Episode Length: {} || Total Reward: {}".format(episode, episodeLength, totalReward))

env.close()

pkl.dump(qTable, open("sarsaQTableModel.pkl", "wb"))
print("Training Complete. Q Table Saved")