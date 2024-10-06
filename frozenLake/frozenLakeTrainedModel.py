import numpy as np
import gym
import numpy as np
import pickle as pkl
import time


env = gym.make("FrozenLake-v1", render_mode="human")


qTable = pkl.load(open("frozenLake\lfrozenLakeQLearningModel.pkl", "rb"))


def policy(state, explore=0.0):
    if(np.random.random() <= explore):
        return np.random.randint(0, 4)  
    else:
        return int(np.argmax(qTable[state]))



NUM_EPISODE = 100
for episode in range(NUM_EPISODE):
    totalReward = 0
    totalEpisode = 0
    done = False
    state, _ = env.reset()

    while not done:
        action = policy(state)
        state, reward, done, truncated, info = env.step(action)


        env.render()

        totalReward += reward
        totalEpisode += 1

    print("EPISODE: {} || TOTAL_EPISODE: {} || TOTAL_REWARD: {}".format(episode, totalEpisode, totalReward))
    
env.close()