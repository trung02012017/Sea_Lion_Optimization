import numpy as np
from copy import deepcopy, copy
import math
import time
import pandas as pd
import os
import json
import random
from fitness_selector import Fitness_Selector


class Modified_Sea_Lion_Optimization(object):

    def __init__(self, fitness_function, dimension, population_size, population, range0, range1, max_ep):

        self.fitness_function = fitness_function
        self.dimension = dimension  # dimension size
        self.population_size = population_size
        self.population = population
        self.best_solution = np.random.uniform(range0, range1, dimension)
        self.best_fitness = sum([self.best_solution[i] ** 2 for i in range(dimension)])
        self.range0 = range0
        self.range1 = range1
        self.max_ep = max_ep

    def get_fitness(self, particle):
        return self.fitness_function(particle)

    def set_best_solution(self):

        fitness = []
        for i in range(self.population_size):
            fitness.append(self.get_fitness(self.population[i]))

        best_index = fitness.index(min(fitness))
        best_solution = self.population[best_index]

        if min(fitness) <= self.get_fitness(self.best_solution):
            self.best_solution = copy(best_solution)

    def crossover(self, agent_1, agent_2):
        feature_index = list(range(self.dimension))
        agent_1_points = random.sample(range(self.dimension), int(self.dimension / 2))
        agent_2_points = [point for point in feature_index if point not in agent_1_points]

        new_agent = np.zeros(self.dimension)
        new_agent[agent_1_points] = agent_1[agent_1_points]
        new_agent[agent_2_points] = agent_2[agent_2_points]

        return new_agent

    def caculate_xichma(self, beta):
        up = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        down = (math.gamma((1 + beta) / 2) * beta * math.pow(2, (beta - 1) / 2))
        xich_ma_1 = math.pow(up / down, 1 / beta)
        xich_ma_2 = 1
        return xich_ma_1, xich_ma_2

    def shrink_encircling_Levy(self, current_whale, epoch_i, dist, c, beta=1):
        xich_ma_1, xich_ma_2 = self.caculate_xichma(beta)
        a = np.random.normal(0, xich_ma_1, 1)
        b = np.random.normal(0, xich_ma_2, 1)
        LB = 0.01 * a / (math.pow(np.abs(b), 1 / beta)) * dist * c
        D = np.random.uniform(self.range0, self.range1, 1)
        levy = LB * D
        return (current_whale - math.sqrt(epoch_i + 1) * np.sign(np.random.random(1) - 0.5)) * levy

    def evaluate_population(self, provs_population):

        # population = np.maximum(population, range0)
        # population = np.minimum(population, range1)
        for i in range(self.population_size):
            for j in range(self.dimension):
                if provs_population[i, j] > self.range1 or float(provs_population[i, j]) == float(self.range1):
                    provs_population[i, j] = np.random.uniform(self.range1 - 2, self.range1)

                if provs_population[i, j] < self.range0 or float(provs_population[i, j]) == float(self.range0):
                    provs_population[i, j] = np.random.uniform(self.range0, self.range0 + 2)

        return provs_population

    def run(self):

        for epoch_i in range(self.max_ep):

            self.set_best_solution()
            print(self.get_fitness(self.best_solution))
            provs_population = np.zeros((self.population_size, self.dimension))

            for i in range(self.population_size):

                current_agent = self.population[i]
                SP_leader = np.random.uniform(0, 1)
                if SP_leader >= 0.6:
                    m = np.random.uniform(-1, 1)
                    new_current_agent = np.abs(self.best_solution - current_agent) * np.cos(2 * np.pi * m) \
                                        + self.best_solution
                else:
                    c = 2 - 2 * epoch_i / self.max_ep
                    b = np.random.uniform(0, 1, self.dimension)
                    p = np.random.uniform(0, 1)

                    if c > 1:
                        a = 0.4
                    else:
                        a = 0.6

                    if p < a:
                        dist = b * np.abs(2 * self.best_solution - current_agent)
                        # new_current_agent = self.best_solution - dist*c
                        #
                        new_current_agent = self.shrink_encircling_Levy(current_agent, epoch_i, dist, c)
                    else:

                        # rand_index = np.random.randint(0, self.population_size)
                        # random_SL = self.population[rand_index]

                        random_SL_1 = self.best_solution

                        rand_index = np.random.randint(0, self.population_size)
                        random_SL_2 = self.population[rand_index]
                        # random_SL = self.crossover(random_SL_1, random_SL_2)
                        random_SL = 2 * random_SL_1 - random_SL_2

                        dist = np.abs(b * random_SL - current_agent)
                        new_current_agent = random_SL - dist*c
                provs_population[i] = new_current_agent
            self.population = copy(self.evaluate_population(provs_population))


def main():

    population_size = 150
    dimension = 30
    range0 = -500
    range1 = 500
    max_ep = 500

    fitness_selector = Fitness_Selector()
    fitness_function = fitness_selector.chose_function('f16')
    population = [np.random.uniform(range0, range1, dimension) for _ in range(population_size)]

    SLnO = Modified_Sea_Lion_Optimization(fitness_function, dimension, population_size, population, range0,
                                          range1, max_ep)
    SLnO.run()


if __name__ == '__main__':
    main()