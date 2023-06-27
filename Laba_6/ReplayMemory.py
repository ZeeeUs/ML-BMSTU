import random
from collections import  deque

from SetUp import Transition

# Реализация техники Replay Memory
class ReplayMemory(object):
  def __init__(self, capacity):
    self.memory = deque([], maxlen=capacity)

  def push(self, *args):
    '''
    Сохранение данных в ReplayMemory
    '''

    self.memory.append(Transition(*args))

  def sample(self, batch_size):
    '''
    Выборка случайных элементов размера batch_size
    '''

    return random.sample(self.memory, batch_size)

  def __len__(self):
    return len(self.memory)