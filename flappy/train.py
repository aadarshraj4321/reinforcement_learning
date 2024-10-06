import torch
import numpy as np
from game import FlappyBirdGame
from agent import Agent
from helper import plot

def train():
    agent = Agent()
    game = FlappyBirdGame()

    scores = []
    mean_scores = []

    while True:
        game.reset()
        state = agent.get_state(game)
        score = 0

        while not game.game_over:
            action = agent.get_action(state)
            reward, done, _ = game.play_step(np.argmax(action))
            next_state = agent.get_state(game)

            agent.train_short_memory(state, action, reward, next_state, done)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            score += reward

            if done:
                break
        
        agent.train_long_memory()

        scores.append(score)
        mean_score = np.mean(scores[-100:])
        mean_scores.append(mean_score)
        plot(scores, mean_scores)

        print(f'Game {agent.n_games} - Score: {score} - Epsilon: {agent.epsilon}')

        agent.n_games += 1

if __name__ == "__main__":
    train()
