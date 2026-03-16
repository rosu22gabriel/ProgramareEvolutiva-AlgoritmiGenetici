
import numpy as np
import random

def calculeaza_fitness(cost, valoare, x, cMax):
    cost_x = np.dot(x, cost)
    valoare_x = np.dot(x, valoare)
    return cost_x<=cMax, valoare_x

def gen_pop(fc, fv, cMax, dim):
    cost = np.genfromtxt("cost.txt", int)
    valoare = np.genfromtxt("valoare.txt", float)
    n = valoare.size
    pop = []
    for i in range(dim):
        gasit = False
        x = []
        while gasit == False:
            x = random.choices([0, 1], k=n)
            gasit, val = calculeaza_fitness(cost, valoare, x, cMax)
        x.append(val)
        pop.append(x)
    return pop

if __name__ == '__main__':
    pop = gen_pop("cost.txt", "valoare.txt", 50, 8)
    for element in pop:
        print(f"Individ: {element[:-1]}")
        print(f"Calitate: {element[-1]}")