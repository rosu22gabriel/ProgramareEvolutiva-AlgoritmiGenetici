import numpy as np
import matplotlib.pyplot as grafic
import copy
import random

from FunctiiSelectii import SUS, elitism
from FunctiiCrossoverIndivizi import crossover_unipunct
from FunctiiMutatieIndivizi import m_binar

def fitness(x, n, m):
    dim = n + m
    cost = 0
    for i in range(dim):
        i0 = i
        i1 = (i + 1) % dim
        i2 = (i + 2) % dim

        if x[i0] == 0 and x[i1] == 1 and x[i2] == 0:
            cost += 1
    suma_x = sum(x)
    cost += abs(m - suma_x)
    return 1 / (1 + cost)


def gen_populatie(dim_pop, n, m):
    dim_total = n + m
    pop = []
    for _ in range(dim_pop):
        x = np.arange(1)
        fit = fitness(x, n, m)
        x.append(fit)
        pop.append(x)
    return pop

def mutatie_populatie(pop, pm, n, m):
    dim_total = n + m
    mpop = copy.deepcopy(pop)
    dim = len(pop)

    for i in range(dim):
        x = pop[i][:n].copy()
        a_avut_loc_o_mutatie = False

        for j in range(n):
            r = random.uniform(0, 1)
            if r <= pm:
                # m_binar face: 0 -> 1 sau 1 -> 0
                x[j] = m_binar(x[j])
                a_avut_loc_o_mutatie = True

        # Daca cel putin un bit a fost schimbat, verificam daca noul individ este valid
        if a_avut_loc_o_mutatie:
            fezabil, val = fitness(x, n, m)
            # Daca mutatia a dus la o solutie valida (nu depaseste greutatea), o pastram
            if fezabil:
                # Reconstruim individul cu noua calitate
                nou_individ = x + [val]
                mpop[i] = nou_individ
            # Nota: Daca mutatia nu e fezabila, individul ramane in starea de dinainte

    return mpop

if __name__ == "__main__":
    n_caini = 4
    m_pisici = 2
    dim_pop = 5
    pm = 0.5
    dim_total = n_caini + m_pisici

    test_gresit = [0, 1, 1, 0, 1, 0]
    rez_test = fitness(test_gresit, n_caini, m_pisici)

    print(rez_test)


    populatie_generata = gen_populatie(dim_pop, n_caini, m_pisici)

    print(populatie_generata)

    # populatie_mutata = mutatie_populatie(populatie_generata, pm, n_caini, m_pisici)

    # print(populatie_mutata)