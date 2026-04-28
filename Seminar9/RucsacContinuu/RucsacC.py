import numpy as np
import matplotlib.pyplot as grafic
import random
import copy

# Importuri din bibliotecile externe
from FunctiiCrossoverIndivizi import crossover_singular, crossover_simplu
from FunctiiMutatieIndivizi import m_neuniforma
from FunctiiSelectii import elitism, ruleta

# 1. Verificarea Fezabilitatii
def ok(x, c, v, cmax):
    """
    Verifica daca un individ respecta restrictia de capacitate a rucsacului
    E: fezabil (Bool), valoare_totala (float)
    """
    val = np.dot(x, v)
    cost = np.dot(x, c)
    return cost <= cmax, val

# 2. Generarea Populatiei Initiale
def gen(c, v, cmax, dim):
    """
    Creeaza o lista de 'dim' indivizi fezabili.
    Fiecare individ este o lista: [gena_1, gena_2, ..., gena_n, fitness]
    """
    n = v.size
    pop = []
    for i in range(dim):
        gata = False
        while not gata:
            x = [random.uniform(0, 1) for _ in range(n)]
            gata, val = ok(x, c, v, cmax)
        x.append(val)
        pop.append(x)
    return x

# 3. Recombinare (Crossover)
def crossover_populatie(pop, c, v, cmax, pc, alpha):
    """
    Realizeaza recombinarea asexuata intre perechi de parinti.
    Foloseste 'crossover_singular" din FunctiiCrossoverIndivizi.
    """
    dim = len(pop)
    n = c.size
    po = copy.deepcopy(pop)
    for i in range(0, dim - 1, 2):
        x = copy.deepcopy(pop[i][:-1])
        y = copy.deepcopy(pop[i+1][:-1])
        if np.random.uniform(0, 1) <= pc:
            # Se creeaza doi copii prin combinarea liniara a unei singure gene
            c1, c2 = crossover_singular(np.array(x), np.array(y), n, alpha)

            # Validam copiii inainte de a-i introduce in populatie
            fez, val = ok(c1, c, v, cmax)
            if fez:
                po[i][:-1], po[i][-1] = c1.tolist(), val
            fez, val = ok(c2, c, v, cmax)
            if fez:
                po[i+1][:-1], po[i+1][-1] = c2.tolist(), val














