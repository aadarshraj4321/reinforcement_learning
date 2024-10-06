import gym
from stable_baselines3 import PPO



env = gym.make("LunarLander-v2", render_mode="human")

#env.reset()

modelsDir = "models/PPO"
modelPath = f"{modelsDir}/550000.zip"

model = PPO.load(modelPath, env=env)

TOTAL_EPISODE = 10
for episode in range(TOTAL_EPISODE):
    done = False
    state, _ = env.reset()

    while not done:
        env.render()
        action, _ = model.predict(state)
        state, reward, done, truncated, info = env.step(action)
        
        #done = done or truncated




env.close()
