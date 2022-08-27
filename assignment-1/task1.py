"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value


# START EDITING HERE
# You can use this space to define any helper functions that you need
def kl(p, q):
    kl_div = np.zeros(np.shape(p))
    zeros = np.where(p == 0)
    ones = np.where(p == 1)
    others = np.flatnonzero(~np.in1d(p, [0, 1]))

    kl_div[zeros] = -np.log(1 - q[zeros])
    kl_div[ones] = -np.log(q[ones])
    kl_div[others] = p[others] * np.log(p[others] / q[others]) + \
        (1 - p[others]) * np.log((1 - p[others]) / (1 - q[others]))

    # if p == 0:
    #     return -math.log(1 - q)
    # elif p == 1:
    #     return -math.log(q)
    # else:
    #     return p * math.log(p / q) + (1 - p) * math.log((1 - p)/(1 - q))
    return kl_div

def compute_kl_ucb(counts, values, t, c=3, N=5, tol=1e-3):
    iter = 0

    lo = np.copy(values)
    hi = np.ones(np.shape(values))

    f = np.ones(np.shape(values))
    kl_ucbs = np.zeros(np.shape(values))
    while (iter < N and np.abs(f).any() > tol):
        kl_ucbs = (lo + hi) / 2
        f = counts * kl(values, kl_ucbs) - math.log(t) - c * math.log(math.log(t))
        above = np.where(f > 0)
        below = np.where(f <= 0)
        lo[below] = kl_ucbs[below]
        hi[above] = kl_ucbs[above]

        iter += 1

    return kl_ucbs
# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.t = 0

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.ucbs = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.t < self.num_arms:
            # First pull every arm once so that quantities are well-defined
            return self.t
        else:
            # Select arm that maximises UCB with random tie-breaking
            return np.argmax(self.ucbs)
            # return np.random.choice(np.flatnonzero(self.ucbs == np.max(self.ucbs)))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

        # Only update UCB after all arms have been sampled exactly once
        if self.t >= self.num_arms:
            self.ucbs = self.values + np.sqrt(2 * math.log(self.t) / self.counts)
            self.ucbs[arm_index] += (new_value - self.values[arm_index])   

        self.t += 1 # Increment global time since start of the algorithm 
        # END EDITING HERE

class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.t = 0

        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
        self.kl_ucbs = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        if self.t < self.num_arms:
            # First pull every arm once so that quantities are well-defined
            return self.t
        else:
            # Select arm that maximises UCB with random tie-breaking
            return np.argmax(self.kl_ucbs)
            # return np.random.choice(np.flatnonzero(self.kl_ucbs == np.max(self.kl_ucbs)))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

        # Only update KL-UCB after all arms have been sampled exactly once
        if self.t >= self.num_arms:
            self.kl_ucbs = compute_kl_ucb(self.counts, self.values, self.t)

        self.t += 1 # Increment global time since start of the algorithm 
        # END EDITING HERE


class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        x = np.random.beta(self.successes + 1, self.failures + 1)
        return np.argmax(x)
        # return np.random.choice(np.flatnonzero(x == np.max(x)))
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        # Depending on reward, either increment successes or failures by 1
        self.successes[arm_index] += reward
        self.failures[arm_index] += (1 - reward)
        pass
        # END EDITING HERE
