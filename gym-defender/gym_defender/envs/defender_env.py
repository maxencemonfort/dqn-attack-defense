import gym
from gym import error, spaces, utils
from gym.utils import seeding
from random import randint
from pulp import *
import os
import time

class Defender(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, K, initial_potential):
        self.state = [0]*(K+1)
        self.K = K
        self.initial_potential = initial_potential
        self.random_start()
        self.done = 0
        self.reward = 0
        self.add = [0, 0]
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


    def optimal_split(self, difference = 0):
        if (sum(self.state) == 1):
            return self.state, [0]*(self.K+1)
        else:
            prob = LpProblem("Optimal split",LpMinimize)
            A = []
            for i in range(self.K + 1):
                A += LpVariable(str(i), 0, self.state[i], LpInteger)
            prob += sum([2**(1-i) * c for c, i in zip(A, range(self.K + 1))]) - self.potential() - difference, "Objective function"
            prob += sum([2**(1-i) * c for c, i in zip(A, range(self.K + 1))]) >= self.potential() + difference, "Constraint"
            prob.writeLP("test.lp")
            prob.solve()
            Abis = [0]*(self.K+1)
            for v in prob.variables():
                Abis[int(v.name)] = round(v.varValue)
            B = [z - a for z, a in zip(self.state, Abis)]
            return Abis, B


    def attacker_play(self):
        prob = 90 #principe de la politique epsilon-greedy
        if(randint(1,100)<=prob):
            return self.optimal_split()
        else:
            return self.optimal_split(difference=randint(1,4)/10)


    def check(self):
        if (sum(self.state) == 0):
            return 1
        elif (self.state[0] >=1 ):
            return 2
        else:
            return 0


    def step(self, target):
        if self.done == 1:
            return [self.state, self.reward, self.done, self.add]
        else:
            if (target == 0):
                self.erase(self.A)
            else:
                self.erase(self.B)
        win = self.check()
        if(win):
            self.done = 1;
            self.add[win-1] = 1;
            if win == 1:
                self.reward = 1
            else:
                self.reward = -1
        else:
            self.A, self.B = self.attacker_play()
        return self.state, self.reward, self.done, self.add


    def reset(self):
        self.state = [0]*(self.K + 1)
        self.random_start()
        self.done = 0
        self.reward = 0
        self.add = [0, 0]
        self.A, self.B = self.attacker_play()
        return self.state


    def render(self):
        for j in range(self.K + 1):
            print(self.state[j], end = " ")
        print("")
