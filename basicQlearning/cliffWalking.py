import gym
import numpy as np
import time


env = gym.make("CliffWalking-v0", render_mode="human")





state = env.reset()
#print(env.observation_space)
done = False

while not done:
    #print(env.render(mode="ansi"))
    action = int(np.random.randint(0, 4, size = 1))
    #print(action)
    state, reward, done, truncated, info = env.step(action)

    done = done or truncated

    env.render()


env.close()