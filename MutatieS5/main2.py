import numpy as np


def genereaza_pop(dim, n):
    pop = np.zeros( [dim, n + 1], dtype="int")
    for i in range(dim):
        pop[i, :n] = np.random.permutation(n)
        pop[i, n] = fitness(pop[i, :n])
    return pop

# x - individ
def fitness(x):
    n = x.size
    # val = len([(i, j) for i in range(n - 1 for j in range( i + 1, n)if x[i] < x[j])
    val = 0
    for i in range(n - 1):
        for j in range( i + 1, n):
            if x[i] < x[j]:
                val += 1
    return val


def mutatie_interschimbare(pop, dim, pm):
    #dim = pop.shape[0] -> .shape intoarce numarul de linii si coloane
    pop_m = pop.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        x = pop_m[i]
        if r <= pm:
            xi = np.random.randint(0, n - 2)
            xj = np.random.randint(xi + 1, n - 1)
            x[xi], x[xj] = x[xj], x[xi]
            x[n] = fitness(x)
            pop_m[i] = x
    return pop_m

if __name__ == "__main__":
    n = 11
    dim = 94
    pm = 0.2
    pop = genereaza_pop(dim, n)
    pop_m = mutatie_interschimbare(pop, dim, pm)
