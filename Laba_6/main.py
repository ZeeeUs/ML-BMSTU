import gymnasium as gym
from DQN_Agent import DQN_Agent

import os
os.environ['SDL_VIDEODRIVER']='dummy'
import pygame
pygame.display.set_mode((640,480))
from SetUp import CONST_ENV_NAME
def main():
        env = gym.make(CONST_ENV_NAME)
        agent = DQN_Agent(env)
        agent.train()
        agent.play_agent()

if __name__ == '__main__':
    main()