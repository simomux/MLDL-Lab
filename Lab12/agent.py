import numpy as np
import random

class Agent:
  """
  Class that models a reinforcement learning agent.
  """

  def __init__(self, n_rows, n_cols, epsilon=0.01, alpha=1, gamma=1):
    self.n_rows = n_rows
    self.n_cols = n_cols

    self.n_actions = 4

    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma

    self.Q = np.random.rand(self.n_rows, self.n_cols, self.n_actions)

  def get_action_eps_greedy(self, r, c):
    """
    Epsilon-greedy sampling of next action given the current state.
    
    Parameters
    ----------
    r: int
      Current `y` position in the labyrinth
    c: int
      Current `x` position in the labyrinth

    Returns
    -------
    action: int
      Action sampled according to epsilon-greedy policy.
    """
    action = 0
    threshold = random.randint(0, 100)/100

    if (threshold >= 1 - self.epsilon):
      action = self.get_action_greedy(r, c)
    else:
      action = random.randint(0, self.n_actions-1)  # Pick action randomly
      print("Picked action randomly:", action)
    
    return action


  def get_action_greedy(self, r, c):
    """
    Greedy sampling of next action given the current state.

    Parameters
    ----------
    r: int
      Current `y` position in the labyrinth
    c: int
      Current `x` position in the labyrinth

    Returns
    -------
    action: int
      Action sampled according to greedy policy.
    """
    max_elem = np.max(self.Q[r,c])  # Return a list with all the scores normalized for each move in that precise position. We then take the max value and we need to get the value on the action axis for the highest score.
    print("Best greedy action:", max_elem)
    list_elem = np.where(self.Q == max_elem)
    print(np.where(self.Q == max_elem))


    return np.where(self.Q == max_elem)[2]

  def update_Q(self, old_state, action, reward, new_state):
    raise NotImplementedError()
