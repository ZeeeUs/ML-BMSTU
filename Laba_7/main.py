import gymnasium as gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from  Policy import Policy
import os
os.environ['SDL_VIDEODRIVER']='dummy'
import pygame
pygame.display.set_mode((640,480))

# Cart Pole
CONST_ENV_NAME = 'Acrobot-v1'
env = gym.make(CONST_ENV_NAME)
GAMMA = 0.99
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

model = Policy()
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
eps = np.finfo(np.float32).eps.item()

def select_action(state):
  state = torch.from_numpy(state).float()
  probs, state_value = model(state)

  # create a categorical distribution over the list of probabilities of actions
  m = Categorical(probs)

  # and sample an action using the distribution
  action = m.sample()

  # save to action buffer
  model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

  # the action to take (left or right)
  return action.item()

def finish_episode():
  """
  Training code. Calculates actor and critic loss and performs backprop.
  """
  R = 0
  saved_actions = model.saved_actions
  policy_losses = [] # list to save actor (policy) loss
  value_losses = [] # list to save critic (value) loss
  returns = [] # list to save the true values

  # calculate the true value using rewards returned from the environment
  for r in model.rewards[::-1]:
    # calculate the discounted value
    R = r + GAMMA * R
    returns.insert(0, R)

  returns = torch.tensor(returns)
  returns = (returns - returns.mean()) / (returns.std() + eps)

  for (log_prob, value), R in zip(saved_actions, returns):
    advantage = R - value.item()

    # calculate actor (policy) loss
    policy_losses.append(-log_prob * advantage)

    # calculate critic (value) loss using L1 smooth loss
    value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

  # reset gradients
  optimizer.zero_grad()

  # sum up all the values of policy_losses and value_losses
  loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

  # perform backprop
  loss.backward()
  optimizer.step()

  # reset rewards and action buffer
  del model.rewards[:]
  del model.saved_actions[:]

def main():
    running_reward = -500

    # run infinitely many episodes
    for i_episode in count(1):
        # print(running_reward)
        # reset environment and episode reward
        state, _ = env.reset()
        ep_reward = 0
        # for each episode, only run 9999 steps so that we don't
        # infinite loop while learning
        for t in range(1, 99999):
            # select action from policy
            action = select_action(state)
            # take the action
            state, reward, done, truncated, _ = env.step(action)
            model.rewards.append(reward)
            ep_reward += reward
            if done or truncated:
                break
        print(ep_reward)
        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # perform backprop
        finish_episode()
        # log results
        if i_episode % 10 == 0:
            print(f"Episode {i_episode}\tLast reward: {ep_reward:.2f}\tAverage reward: {running_reward:.2f}")
        # check if we have "solved" the cart pole problem
        if running_reward > env.spec.reward_threshold * 2:
            print(f"Solved! Running reward is now {running_reward} and the last episode runs to {t} time steps!")
            break
    env2 = gym.make(CONST_ENV_NAME, render_mode='human')
    # reset environment and episode reward
    state, _ = env2.reset()
    ep_reward = 0
    # for each episode, only run 9999 steps so that we don't
    # infinite loop while learning
    for t in range(1, 10000):
        # select action from policy
        action = select_action(state)
        # take the action
        state, reward, done, _, _ = env2.step(action)
        model.rewards.append(reward)
        ep_reward += reward
        if done:
            break
if __name__ == '__main__':
    main()