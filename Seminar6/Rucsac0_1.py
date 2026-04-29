import numpy as np
import random
import matplotlib.pyplot as grafic
import copy
from typing import List, Tuple, Union

# --- Enunt Cerinta ---
# Problema Rucsacului 0/1:
# Avem n obiecte, fiecare cu un cost si o valoare. Trebuie sa alegem un subset
# de obiecte astfel incat suma costurilor sa nu depaseasta Cmax,
# iar valoarea totala sa fie maximizata.
# Reprezentare: Vector binar (0 = obiect neales, 1 = obiect ales).
# ----------------------

def calculeaza_fitness(cost: np.ndarray, valoare: np.ndarray, x: Union[List[int], np.ndarray], Cmax: float) -> Tuple[bool, float]:
    """
    Evalueaza o solutie binara.

    Args:
        cost: Vectorul cu costurile fiecarui obiect.
        valoare: Vectorul cu valorile fiecarui obiect.
        x: Genotipul (vector binar).
        Cmax: Capacitatea limita.
    Returns:
        (fezabil, valoare_totala): Daca solutia respecta Cmax si ce profit aduce.
    """
    # np.dot realizeaza suma produselor x[i]*cost[i], eficient pentru vectori binari
    cost_x = np.dot(x, cost)
    valoare_x = np.dot(x, valoare)
    return cost_x <= Cmax, valoare_x

def mutatie_pop(pop: List[List[Union[int, float]]], cost: np.ndarray, valoare: np.ndarray, Cmax: float, pm: float) -> List[List[Union[int, float]]]:
    """
    Aplica mutatia Bit-Flip: fiecare bit poate fi inversat (0 -> 1, 1 -> 0) cu probabilitatea pm.
    """
    pop_m = copy.deepcopy(pop)
    dim = len(pop)
    n = cost.size

    for i in range(dim):
        # Extragem doar genele (fara ultima coloana care e fitness-ul)
        x = pop_m[i][:n].copy()
        mutat = False
        for j in range(n):
            if random.uniform(0, 1) <= pm:
                # Inversare bit: int(not 1) -> 0, int(not 0) -> 1
                x[j] = int(not x[j])
                mutat = True
        if mutat:
            # Dupa mutatie, verificam daca rucsacul mai este valid (fezabil)
            fezabil, val = calculeaza_fitness(cost, valoare, x, Cmax)
            if fezabil:
                # Daca e valid, actualizam individul si valoarea sa in populatie
                nou_individ = list(x) + [val]
                pop_m[i] = nou_individ
            # Daca nu e fezabil, individul ramane cel nemutat din pop_m[i] (parintele)

    return pop_m

def crossover_uniform(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Operator Crossover Uniform: Pentru fiecare pozitie, copiii primesc genele
    de la unul din parinti cu probabilitatea 50%.
    """
    c1 = x.copy()
    c2 = y.copy()
    for i in range(n):
        # Aruncam cu banul (0 sau 1)
        if np.random.randint(0, 2) == 1:
            # Interschimbam genele intre cei doi copii
            c1[i], c2[i] = y[i], x[i]
    return c1, c2


def crossover_populatie(pop: List[List[Union[int, float]]], cost: np.ndarray, valoare: np.ndarray, Cmax: float, pc: float) -> List[List[Union[int, float]]]:
    """
    Gestioneaza recombinarea la nivelul intregii populatii.
    """
    dim = len(pop)
    n = cost.size
    po = copy.deepcopy(pop)

    # Parcurgem populatia in perechi (0-1, 2-3...)
    for i in range(0, dim - 1, 2):
        parinte_x = np.array(pop[i][:-1])
        parinte_y = np.array(pop[i + 1][:-1])

        if np.random.uniform(0, 1) <= pc:
            c1, c2 = crossover_uniform(parinte_x, parinte_y, n)

            # Validam copilul 1
            fez1, val1 = calculeaza_fitness(cost, valoare, c1, Cmax)
            if fez1:
                po[i] = list(c1) + [val1]

            # Validam copilul 2
            fez2, val2 = calculeaza_fitness(cost, valoare, c2, Cmax)
            if fez2:
                po[i+1] = list(c2) + [val2]
    return po


def genereaza_populatie(cost: np.ndarray, valoare: np.ndarray, Cmax: float, dim: int) -> List[List[Union[int, float]]]:
    """
    Creeaza populatia initiala asigurandu-se ca toti indivizii sunt fezabili.
    """
    n = valoare.size
    pop = []
    for i in range(dim):
        gasit = False
        while not gasit:
            # Generam un vector binar aleator de lungime n
            x = random.choices([0, 1], k=n)
            gasit, val_fit = calculeaza_fitness(cost, valoare, x, Cmax)

        x.append(val_fit)
        pop.append(x)
    return pop

if __name__ == "__main__":
    # Parametri test
    CAPACITATE_MAX = 50
    DIM_POP = 10
    P_MUTATIE = 0.2
    P_CROSSOVER = 0.8

    # Incercare citire fisiere
    try:
        cost_obiecte = np.genfromtxt("cost_disc.txt")
        valoare_obiecte = np.genfromtxt("valoare_disc.txt")
    except:
        print("Fisierele nu exista. Generam date de test...")
        cost_obiecte = np.array([10, 20, 30, 15, 25])
        valoare_obiecte = np.array([100, 200, 150, 180, 220])

    # 1. Initializare
    populatie = genereaza_populatie(cost_obiecte, valoare_obiecte, CAPACITATE_MAX, DIM_POP)
    print(f"Cel mai bun Initial: {max(populatie, key=lambda x: x[-1])}")

    # 2. Mutatie
    pop_dupa_mutatie = mutatie_pop(populatie, cost_obiecte, valoare_obiecte, CAPACITATE_MAX, P_MUTATIE)
    print(f"Cel mai bun dupa Mutatie: {max(pop_dupa_mutatie, key=lambda x: x[-1])}")

    # 3. Crossover
    pop_dupa_crossover = crossover_populatie(populatie, cost_obiecte, valoare_obiecte, CAPACITATE_MAX, P_CROSSOVER)
    print(f"Cel mai bun dupa Recombinare: {max(pop_dupa_crossover, key=lambda x: x[-1])}")
















