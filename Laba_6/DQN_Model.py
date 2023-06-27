import torch.nn as nn
import torch.nn.functional as F

class DQN_Model(nn.Module):
  def __init__(self, n_observations, n_actions):
    '''
    Инициализация топологии нейронной сети
    '''

    super(DQN_Model, self).__init__()
    self.layer1 = nn.Linear(n_observations, 128)
    self.layer2 = nn.Linear(128, 64)
    self.layer3 = nn.Linear(64, n_actions)

  def forward(self, x):
    '''
    Прямой проход
    Вызывается для одного элемента, чтобы определить следующее действие
    Или для batch во время процедуры оптимизации
    '''

    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)