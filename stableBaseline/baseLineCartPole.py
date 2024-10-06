import gym
import numpy as np
from stable_baselines3 import PPO




env = gym.make("CartPole-v1", render_mode="human")


modelsDir = "modelsCartPole/PPO"
logdir = "logsCartPole"


if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
if not os.path.exists(logdir):
    os.makedirs(logdir)


model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
# model = A2C("MlpPolicy", env, verbose=1)



TOTAL_EPISODE = 10
for episode in range(TOTAL_EPISODE):
    done = False
    state, _ = env.reset()
    while not done:
        state, reward, done, truncated, info = env.step(env.action_space.sample())

        TIMESTEPS = 10000
        for i in range(1, 100):
            model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
            model.save(f"{modelsDir}/{TIMESTEPS * i}")



        env.render()
    
env.close()