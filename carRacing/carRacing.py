import numpy as np
import gym

# Create the CarRacing environment
env = gym.make("CarRacing-v2", render_mode="human")


def policy(state, explore=0.0):
    if(np.random.random() <= explore):
        return 



# Parameter
NUM_EPISODE = 5
EPSILON = 0.1
GAMMA = 0.9


for i in range(len(NUM_EPISODE)):
    done = False
    state, _ = env.reset()


    while not done:
        # Random action (steering, acceleration, braking)
        action = np.array([np.random.uniform(-1, 1), np.random.uniform(1, 1), np.random.uniform(0, 0.3)])
        
        
        state, reward, done, truncated, info = env.step(action)
        
    
        env.render()


env.close()
