import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint, choice
from pulp import *
import os
import time

def potential(A):
        potential = 0
        for i in range(len(A)):
            potential += A[i] * 2**(-i)
        return potential

class Defender(gym.Env):
    #metadata = {'render.modes': ['human']}


    def __init__(self, K, initial_potential):
        self.state = [0]*(K+1)
        self.K = K
        self.initial_potential = initial_potential
        self.random_start()
        self.done = 0
        self.reward = 0
        self.A, self.B = self.attacker_play()


    def random_start(self):
        potential = self.potential()
        stop = False
        while (potential < self.initial_potential and not stop):
            possible = self.initial_potential - potential
            upper = 1 #upper is 1 because 0 represents the top of the matrix which means end of the game
            while (2**(-upper) > possible):
                upper +=1
            if(upper > self.K):
                stop = True
            else:
                self.state[randint(upper,self.K)]+=1
                potential = self.potential()


    def potential(self):
        potential = 0
        for i in range(self.K + 1):
            potential += self.state[i] * 2**(-i)
        return potential


    def subpotential(self):
        potentialA = 0
        potentialB = 0
        for i in range(self.K + 1):
            potentialA += self.A[i] * 2**(-i)
            potentialB += self.B[i] * 2**(-i)
        return potentialA, potentialB


    def split(self, A):
        B = [z - a for z, a in zip(self.state, A)]
        return A, B


    def erase(self, A):
        self.state = [z - a for z, a in zip(self.state, A)]
        self.state = self.state[1:] + [0]


    def optimal_split(self, ratio = 0.5):
        if (sum(self.state) == 1):
            return self.state, [0]*(self.K+1)
        else:
            A = [0] * (self.K + 1)
            B = [0] * (self.K + 1)
            l = 0
            while (potential(A) < ratio * self.potential() and l < self.K + 1):
                A[l] = self.state[l]
                l += 1
            for j in range(l, self.K + 1):
                B[j] = self.state[j]
            if potential(A) == ratio * self.potential():
                return A,B
            elif potential(A) > ratio * self.potential():
                while (potential(A) > ratio * self.potential()):
                    B[l - 1] += 1
                    A[l - 1] -= 1
                difference = ratio * self.potential() - potential(A)
                if (difference > 2**(- 1 - l)):
                    B[l - 1] -= 1
                    A[l - 1] += 1
                return A, B
            else:
                print('error')
                return None


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def attacker_play(self):
        prob = 90 #principe de la politique epsilon-greedy
        if(randint(1,100)<=prob):
            return self.optimal_split()
        else:
            ratios = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
            return self.optimal_split(ratio=choice(ratios))


    def check(self):
        if (sum(self.state) == 0):
            return 1
        elif (self.state[0] >=1 ):
            return 2
        else:
            return 0


    def step(self, target):
        if self.done == 1:
            return self.A + self.B, self.reward, self.done, {}
        else:
            if (target == 0):
                self.erase(self.A)
            else:
                self.erase(self.B)
        win = self.check()
        if(win):
            self.done = 1;
            if win == 1:
                self.reward = 1
            else:
                self.reward = -1
        self.A, self.B = self.attacker_play()
        return self.A + self.B, self.reward, self.done, {}


    def reset(self):
        self.state = [0]*(self.K + 1)
        self.random_start()
        self.done = 0
        self.reward = 0
        self.add = [0, 0]
        self.A, self.B = self.attacker_play()
        return self.A + self.B


    def _get_obs(self):
        return self.state


    def render(self):
        for j in range(self.K + 1):
            print(self.state[j], end = " ")
        print("")
