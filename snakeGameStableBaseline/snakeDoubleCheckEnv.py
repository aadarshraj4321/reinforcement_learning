from snakeCheckEnv import SnakeEnv



env = SnakeEnv()

episodes = 50

for episode in range(episodes):
    done = False
    state, _ = env.reset()

    while not done:
        random_action = env.action_space.sample()
        print("Action: {}".format(random_action))
        state, reward, done, truncated, info = env.step(random_action)
        print("Reward: {}".format(reward))
        