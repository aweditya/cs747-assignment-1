"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the AlgorithmManyArms class. Here are the method details:
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
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need
# END EDITING HERE

class AlgorithmManyArms:
    def __init__(self, num_arms, horizon):
        self.k = int(np.sqrt(num_arms))

        self.num_arms = num_arms
        arms = np.arange(0, num_arms)
        np.random.shuffle(arms)
        self.arms = arms[0:self.k]

        self.mapping = dict()
        for index in range(self.num_arms):
            self.mapping[arms[index]] = index

        self.successes = np.zeros(self.k)
        self.failures = np.zeros(self.k)
        # Horizon is same as number of arms
    
    def give_pull(self):
        # START EDITING HERE
        x = np.random.beta(self.successes + 1, self.failures + 1)
        return self.arms[np.argmax(x)]
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.successes[self.mapping[arm_index]] += reward
        self.failures[self.mapping[arm_index]] += (1 - reward)
        # END EDITING HERE
