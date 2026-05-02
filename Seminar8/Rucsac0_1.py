import numpy as np
import random
import matplotlib.pyplot as grafic
import copy
from typing import List, Tuple, Union

# =============================================================================
# 1. EVALUARE ȘI INIȚIALIZARE (Reprezentare Binară)
# =============================================================================

def calculeaza_fitness(cost: np.ndarray, valoare: np.ndarray, x: List[int], Cmax: float) -> Tuple[bool, float]:
    """
    Evalueaza o solutie binara (0/1).
    Calculul se face prin produsul scalar (dot product) intre vectorul de selectie si cel de atribute.

    Args:
        x: Genotipul (vector binar de lungime n).
    Returns:
        (fezabil, profit): True daca greutatea <= Cmax, si profitul total generat.
    """
    cost_x = np.dot(x, cost)
    valoare_x = np.dot(x, valoare)
    return cost_x <= Cmax, float(valoare_x)

def genereaza_populatie(cost: np.ndarray, valoare: np.ndarray, Cmax: float, dim: int) -> List[List[Union[int, float]]]:
    """
    Genereaza o populatie de dimensiune 'dim' formata din indivizi binari aleatori.
    Se asigura prin incercari repetate ca fiecare individ initial este fezabil.
    """
    n = valoare.size
    pop = []
    for i in range(dim):
        gasit = False
        x = []
        while not gasit:
            # random.choices genereaza un vector de lungime n cu valori de 0 sau 1
            x = random.choices([0, 1], k = n)
            gasit, val = calculeaza_fitness(cost, valoare, x, Cmax)
        # Structura individului: [gene_0, ..., gene_n, fitness)
        x.append(val)
        pop.append(x)
    return pop

# =============================================================================
# 2. OPERATORI DE VARIAȚIE (Bit-Flip & Crossover Uniform)
# =============================================================================
def mutatie_pop(pop: List[List[Union[int, float]]], cost: np.ndarray, valoare: np.ndarray,
                Cmax: float, pm: float) -> List[List[Union[int, float]]]:
    """
    Mutatie Bit-Flip: Pentru fiecare bit, exista o probabilitate 'pm' de a fi inversat.
    Daca rezultatul mutatie incalca restrictia Cmax, se pastreaza individul original.
    """
    pop_m = copy.deepcopy(pop)
    dim = len(pop)
    n = cost.size()

    for i in range(dim):
        x = pop_m[i][:n].copy()
        mutat = False
        for j in range(n):
            r = random.uniform(0, 1)
            if r <= pm:
                x[j] = int(not x[j]) # Inversare 0 -> 1 sau 1 -> 0
                mutat = True
        if mutat:
            gasit, val = calculeaza_fitness(cost, valoare, x, Cmax)
            if gasit:
                # Daca noua configuratie este valida, actualizam fitness-ul
                x_nou = x + [val]
                pop_m[i] = x_nou
    return pop_m


def crossover_uniform(x: List[int], y: List[int], n: int) -> Tuple[List[int], List[int]]:
    """
    Crossover Uniform: Pentru fiecare pozitie, se decide aleatoriu (50-50)
    daca genele parintilor sunt interschimbate sau nu.
    """
    c1, c2 = x.copy(), y.copy()
    for i in range(n):
        # r este 0 sau 1 cu probabilitate egala
        r = np.random.randint(0, 2)
        if r == 1:
            c1[i], c2[i] = y[i], x[i]
    return c1, c2

def crossover_populatie(pop: List[List[Union[int, float]]], cost: np.ndarray,
                        valoare: np.ndarray, Cmax: float, pc: float) -> List[List[Union[int, float]]]:
    """
    Aplica recombinarea uniforma asupra populatiei.
    Indivizii sunt selectati in perechi (0, 1), (2, 3) etc.
    """
    dim = len(pop)
    n = cost.size
    po = copy.deepcopy(pop)
    for i in range(0, dim - 1, 2):
        x = pop[i][:-1].copy()
        y = pop[i + 1][: -1].copy()
        if np.random.uniform(0, 1) <= pc:
            c1, c2 = crossover_uniform(x, y, n)

            # Verificam fezabilitatea pentru primul copil
            fez1, val1 = calculeaza_fitness(cost, valoare, c1, Cmax)
            if fez1:
                po[i][:-1], po[i][-1] = c1, val1

            # Verificam fezabilitatea pentru al doilea copil
            fez2, val2 = calculeaza_fitness(cost, valoare, c2, Cmax)
            if fez2:
                po[i + 1][:-1], po[i+1][-1] = c2, val2
    return po

# =============================================================================
# 3. OPERATORI DE SELECȚIE
# =============================================================================
def turneu(pop: np.ndarray, qual: np.ndarray, dim: int, n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selectia Turneu: Selecteaza cel mai bun individ dintr-un grup aleatoriu de 'k' membri.
    """
    spop = np.zeros((dim, n))
    squal = np.zeros(dim)
    for it in range(dim):
        poz = np.random.randint(0, len(pop), k)
        v = [qual[p] for p in poz]
        imax = np.argmax(v)
        isel = poz[imax]

        spop[it][:] = pop[isel][:]
        squal[it] = qual[isel]
    return spop, squal

def elitism(pop_c: List[List[float]], qual_c: List[float], pop_mo: List[List[float]],
            qual_mo: List[float], dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asigura conservarea celui mai bun individ (elitist) din generatia curenta in generatia mutata/recombinata.
    """
    pop_rez = np.array(pop_mo)
    qual_rez = np.array(qual_mo)

    if np.max(qual_c) > np.max(qual_mo):
        imax = np.argmax(qual_c)
        ir = np.random.randint(dim)
        pop_rez[ir] = np.array(pop_c[imax].copy())
        qual_rez[ir] = qual_c[imax]
    return pop_rez, qual_rez

# =============================================================================
# EXECUȚIE
# =============================================================================
if __name__ == "__main__":
    # Configuratii
    C_MAX = 50
    DIM_POP = 10
    PM = 0.2
    PC = 0.8

    # Incarcare date (Simulat)
    cost = np.array([12, 15, 20, 8, 10])
    valoare = np.array([20, 30, 45, 15, 25])

    # Rularea algoritmului (o generatie)
    pop_initiala = genereaza_populatie(cost, valoare, C_MAX, DIM_POP)
    pop_mutata = mutatie_pop(pop_initiala, cost, valoare, C_MAX, PM)
    pop_recombinata = crossover_populatie(pop_initiala, cost, valoare, C_MAX, PC)

    # Exemplu de selectie turneu
    genotipuri = np.array([ind[:-1] for ind in pop_initiala])
    calitati = np.array([ind[-1] for ind in pop_initiala])
    parinti, fitness_p = turneu(genotipuri, calitati, DIM_POP, cost.size, 2)

    print(f"Cel mai bun profit gasit: {max(calitati)}")