import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint, choice
import numpy as np


class Attacker(gym.Env):


    def __init__(self, K, initial_potential):
        self.state = None
        self.K = K
        self.initial_potential = initial_potential
        self.weights = np.power(2.0, [-(self.K - i) for i in range(self.K + 1)])
        self.done = 0
        self.reward = 0
        self.action_space = spaces.Discrete(self.K + 1)
        self.observation_space= spaces.MultiDiscrete([10]* (K+1))
        

    def potential(self, A):
        return np.sum(A*self.weights)


    def split(self, A):
        B = [z - a for z, a in zip(self.state, A)]
        return A, B


    def erase(self, A):
        """Function to remove the partition A from the game state

        Arguments:
            A {list} -- The list representing the partition we want to remove
        """

        self.state = [z - a for z, a in zip(self.state, A)]
        self.state = [0] + self.state[:-1] 


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def defense_play(self, A, B):
        potA = self.potential(A)
        potB = self.potential(B)
        if (potA >= potB):
            self.erase(A)
        else:
            self.erase(B)


    def check(self):
        """Function to chek if the game is over or not.

        Returns:
            int -- If the game is not over returns 0, otherwise returns -1 if the defender won or 1 if the attacker won.
        """

        if (sum(self.state) == 0):
            return -1
        elif (self.state[-1] >=1 ):
            return 1
        else:
            return 0


    def step(self, target):
        A = [0] * (self.K + 1)
        B = [0] * (self.K + 1)
        for i in range(target):
            A[i] = self.state[i]
        for i in range(target + 1, self.K + 1):
            B[i] = self.state[i]
        n = self.state[target]
        while (n>0):
            if self.potential(A) > self.potential(B):
                B[target] += 1
            else:
                A[target] += 1
            n -= 1
        self.defense_play(A,B)
        win = self.check()
        if(win):
            self.done = 1
            self.reward = win

        return self.state, self.reward, self.done, {}


    def reset(self):
        self.state = self.random_start()
        self.done = 0
        self.reward = 0
        return self.state

    def random_start(self):
        self.state = [0] * (self.K + 1)
        potential = 0
        stop = False
        while (potential < self.initial_potential and not stop):
            possible = self.initial_potential - potential
            upper = self.K - 1 # upper is K-1 because K represents the top of the matrix which means end of the game
            while (2**(-(self.K-upper)) > possible):
                upper -=1
            if(upper < 0):
                stop = True
            else:
                self.state[randint(0,upper)]+=1
                potential = self.potential(self.state)
        return self.state


    def render(self):
        for j in range(self.K + 1):
            print(self.state[j], end = " ")
        print("")
