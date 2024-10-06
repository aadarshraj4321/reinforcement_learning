import gym
#from stable_baselines3 import A2C
from stable_baselines3 import PPO
import os


modelsDir = "models/PPO"
logdir = "logs"


if not os.path.exists(modelsDir):
    os.makedirs(modelsDir)
if not os.path.exists(logdir):
    os.makedirs(logdir)



env = gym.make("LunarLander-v2")
#env.reset()

# print("Sample Action: {}".format(env.action_space.sample()))
# print("Observation Space Shape: {}".format(env.observation_space.shape))
# print("Sample Obervation: {}".format(env.observation_space.sample()))



model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
# model = A2C("MlpPolicy", env, verbose=1)
TIMESTEPS = 10000
for i in range(1, 100):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{modelsDir}/{TIMESTEPS * i}")





# for ep in range(episodes):
#     state, _ = env.reset()
#     done = False
#     while not done:
       
#         state, reward, done, truncated, info = env.step(env.action_space.sample())
#         #print(reward)

#         env.render()

env.close()