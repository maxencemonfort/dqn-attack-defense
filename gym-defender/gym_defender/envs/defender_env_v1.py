import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint, choice
import numpy as np
from pulp import *


class Defenderv1(gym.Env):


    def __init__(self, K, initial_potential):
        self.state = None
        self.game_state = None
        self.K = K
        self.initial_potential = initial_potential
        self.weights = np.power(2.0, [-(self.K - i) for i in range(self.K + 1)])
        self.done = 0
        self.reward = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space= spaces.MultiDiscrete([10]* (2*K+2))
        

    def potential(self, A):
        return np.sum(A*self.weights)


    def split(self, A):
        B = [z - a for z, a in zip(self.game_state, A)]
        return A, B


    def erase(self, A):
        """Function to remove the partition A from the game state

        Arguments:
            A {list} -- The list representing the partition we want to remove
        """

        self.game_state = [z - a for z, a in zip(self.game_state, A)]
        self.game_state = [0] + self.game_state[:-1] 


    def optimal_split(self, ratio = 0.5):
        """Function that returns the optimal split for a certain ratio of the potential (default to 0.5)

        Keyword Arguments:
            ratio {float} -- The ratio of the potential needed (default: {0.5})

        Returns:
            list tuple -- Returns the tuple (A, B) representing the partitions.
        """

        if (sum(self.game_state) == 1):
            if (randint(1,100)<=50):
                return self.game_state, [0]*(self.K+1)
            else:
                return [0]*(self.K+1), self.game_state
            
        else:
            prob = LpProblem("Optimal split",LpMinimize)
            A = []
            for i in range(self.K + 1):
                A += LpVariable(str(i), 0, self.game_state[i], LpInteger)
            prob += sum([2**(-(self.K - i)) * c for c, i in zip(A, range(self.K + 1))]) - ratio * self.potential(self.game_state), "Objective function"
            prob += sum([2**(-(self.K - i)) * c for c, i in zip(A, range(self.K + 1))]) >= ratio * self.potential(self.game_state), "Constraint"
            prob.writeLP("test.lp")
            prob.solve()
            Abis = [0]*(self.K+1)
            for v in prob.variables():
                Abis[int(v.name)] = round(v.varValue)
            B = [z - a for z, a in zip(self.state, Abis)]
            return Abis, B


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def attacker_play(self):
        prob = 90 
        if(randint(1,100)<=prob):
            return self.optimal_split()
        else:
            ratios = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
            return self.optimal_split(ratio=choice(ratios))


    def check(self):
        """Function to chek if the game is over or not.

        Returns:
            int -- If the game is not over returns 0, otherwise returns 1 if the defender won or -1 if the attacker won.
        """

        if (sum(self.game_state) == 0):
            return 1
        elif (self.game_state[-1] >=1 ):
            return -1
        else:
            return 0


    def step(self, target):
        A = self.state[: self.K + 1]
        B = self.state[self.K + 1 :]
        if (target == 0):
            self.erase(A)
        else:
            self.erase(B)
        win = self.check()
        if(win):
            self.done = 1
            self.reward = win

        if self.done != 1:
            A, B = self.attacker_play()
            self.state = np.concatenate([A,B])

        return self.state, self.reward, self.done, {}


    def reset(self):
        self.game_state = self.random_start()
        self.done = 0
        self.reward = 0
        A, B = self.attacker_play()
        self.state = np.concatenate([A,B])
        return self.state

    def random_start(self):
        self.game_state = [0] * (self.K + 1)
        potential = 0
        stop = False
        while (potential < self.initial_potential and not stop):
            possible = self.initial_potential - potential
            upper = self.K - 1 #upper is K-1 because K represents the top of the matrix which means end of the game
            while (2**(-(self.K-upper)) > possible):
                upper -=1
            if(upper < 0):
                stop = True
            else:
                self.game_state[randint(0,upper)]+=1
                potential = self.potential(self.game_state)
        return self.game_state


    def render(self):
        for j in range(self.K + 1):
            print(self.game_state[j], end = " ")
        print("")
