import pygame
import random
import numpy as np
from enum import Enum

pygame.init()

# Define constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
PIPE_WIDTH = 70
PIPE_HEIGHT = 500
PIPE_GAP = 150
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
GRAVITY = 0.5
FLAP_STRENGTH = -10
SPEED = 4

class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT
        self.velocity = 0
        self.score = 0

    def flap(self):
        self.velocity = FLAP_STRENGTH

    def move(self):
        self.velocity += GRAVITY
        self.y += self.velocity

class Pipe:
    def __init__(self, x):
        self.x = x
        self.height = random.randint(150, 400)
        self.width = PIPE_WIDTH

    def move(self):
        self.x -= SPEED

class FlappyBirdGame:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('Flappy Bird')
        self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH + 100)]
        self.game_over = False

    def reset(self):
        self.bird = Bird()
        self.pipes = [Pipe(SCREEN_WIDTH + 100)]
        self.game_over = False

    def is_collision(self):
        if self.bird.y > SCREEN_HEIGHT - self.bird.height or self.bird.y < 0:
            return True
        
        for pipe in self.pipes:
            if (pipe.x < self.bird.x + self.bird.width and
                pipe.x + pipe.width > self.bird.x and
                (self.bird.y < pipe.height - PIPE_GAP or
                 self.bird.y + self.bird.height > pipe.height)):
                return True

        return False

    def play_step(self, action):
        if self.game_over:
            return -10, True, self.bird.score
        
        self.bird.flap() if action == 1 else None
        self.bird.move()
        self.update_pipes()
        
        if self.is_collision():
            self.game_over = True
            return -10, True, self.bird.score
        
        reward = 0
        if self.pipes[0].x + self.pipes[0].width < self.bird.x and not self.pipes[0].x == -1:
            self.bird.score += 1
            reward = 10
            self.pipes.pop(0)
        
        self.update_ui()
        return reward, self.game_over, self.bird.score

    def update_pipes(self):
        for pipe in self.pipes:
            pipe.move()
        if self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe(SCREEN_WIDTH))

    def update_ui(self):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 0), pygame.Rect(self.bird.x, self.bird.y, self.bird.width, self.bird.height))
        for pipe in self.pipes:
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(pipe.x, 0, pipe.width, pipe.height))
            pygame.draw.rect(self.screen, (0, 255, 0), pygame.Rect(pipe.x, pipe.height + PIPE_GAP, pipe.width, SCREEN_HEIGHT))
        pygame.display.flip()
        self.clock.tick(SPEED)
