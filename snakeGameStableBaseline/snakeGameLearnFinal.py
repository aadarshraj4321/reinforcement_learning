import gym
from stable_baselines3 import PPO
import os
import time
from snakeCheckEnv import SnakeEnv


modelsDir = f"snakeModels/PPO"
logDir = f"snakeLogs"


if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)

if not os.path.exists(logDir):
    os.makedirs(logDir)


env = SnakeEnv()
env.reset()


model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log=logDir)

TIMESTEPS = 10000
for i in range(1, 10000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{modelsDir}/{TIMESTEPS * i}")

# TOTAL_EPISODE = 10
# for episode in range(TOTAL_EPISODE):
#     state, _ = env.reset()
#     done = False

#     while not done:
#         env.render()
#         state, reward, done, truncated, info = env.step(env.action_space.sample())
#         TIMESTEPS = 10000
#         for i in range(1, 10):
#             model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
#             model.save(f"{modelsDir}/{TIMESTEPS}")
#         env.render()

env.close()