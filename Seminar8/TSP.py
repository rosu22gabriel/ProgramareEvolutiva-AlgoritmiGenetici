"""
TSP (Traveling Salesman Problem)
In aceasta problema, genotipul este o permutare a oraselor, iar
obiectivul este minimizarea costului total al drumului (ceea ce corespunde
maximizarii valorii fitness, calculata ca 100/cost)
"""
import copy
import numpy as np
import random
import matplotlib.pyplot as grafic
from typing import List, Tuple, Union

# =============================================================================
# 1. EVALUARE ȘI INIȚIALIZARE (Problema Comis-Voiajorului)
# =============================================================================
def functie_fitness(cost: np.ndarray, p: np.ndarray) -> float:
    """
    Calculeaza calitatea unui drum (permutare).
    Fitness = 100 / (Suma costurilor intre orase consecutive + revenirea in orasul de start)
    """
    cost_p = 0
    n = len(p)
    for i in range(n - 1):
        cost_p += cost[p[i], p[i + 1]]
    # Se adauga costul de la ultimul oras inapoi la primul oras din permutare
    cost_p += cost[p[0], p[n - 1]]
    return 100 / cost_p

def genereaza_populatie(fc: str, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Incarca matricea de costuri si genereaza o populatie initiala de permutari aleatorii.
    """
    cost = np.genfromtxt(fc)
    n = len(cost)
    pop = np.zeros([dim, n], dtype=int)
    val_pop = np.zeros(dim, dtype=float)
    for i in range(dim):
        pop[i] = np.random.permutation(n)
        val_pop[i] = functie_fitness(cost, pop[i])
    return pop, val_pop, cost

# =============================================================================
# 2. OPERATORI DE VARIAȚIE (Inversiune & PMX)
# =============================================================================
def mutatie_inversiune(p: np.ndarray) -> np.ndarray:
    """
    Mutatie prin Inversiune: Alege doua puncte si inverseaza ordinea genelor intre ele.
    Este un operator foarte eficient pentru TSP, pastrand adiacenta genelor.
    """
    n = p.size
    i = np.random.randint(0, n - 1)
    j = np.random.randint(i + 1, n)
    r = p.copy()
    # Inversarea secventei intre indicii i si j
    r[i:j+1] = p[i:j+1][::-1]
    return r

def mutatie_populatie(pop: np.ndarray, val_pop: np.ndarray, cost: np.ndarray, pm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica mutatia prin inversiune asupra populatiei cu probabilitatea pm."""
    pop_m = pop.copy()
    val_pop_m = val_pop.copy()
    dim = pop_m.shape[0]
    for i in range(dim):
        if np.random.uniform(0, 1) <= pm:
            y = mutatie_inversiune(pop_m[i])
            pop_m[i] = y
            val_pop_m[i] = functie_fitness(cost, y)
    return pop_m, val_pop_m

def PMX(x: np.ndarray, y: np.ndarray, p1:int, p2: int) -> np.ndarray:
    """
    Partially Mapped Crossover (PMX): Construieste un copil pastrand o secveneta
    din x si complemtand restul din y, rezolvand conflictele de tip duplicat
    """
    n = x.size
    copil = -np.ones(n, dtype="int")
    # Copiem secventa de crossover din primul parinte
    copil[p1:p2+1] = x[p1:p2+1]

    # Tratarea alelelor din secventa parintelui y care nu sunt deja in copil
    poz_alele_neplasate = [k for k in range(p1, p2 + 1) if not y[k] in copil]
    for p in poz_alele_neplasate:
        a = y[p]
        curr_p = p
        # Cautam o pozitie libera folosind maparea din secventa de crossover
        while copil[curr_p] > -1:
            val_in_copil = copil[curr_p]
            curr_p = np.where(y == val_in_copil)[0][0]
        copil[curr_p] = a

    # Completam restul pozitiilor ramase cu genele din y
    alele_neplasate = [gene for gene in y if gene not in copil]
    pozitii_libere = [k for k in range(n) if copil[k] == -1]
    copil[pozitii_libere] = alele_neplasate
    return copil

def crossover_PMX(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Alege doua puncte de taiere s i genereaza doi copii prin PMX."""
    n = x.size
    p1 = np.random.randint(0, n - 1)
    p2 = np.random.randint(p1 + 1, n)
    return PMX(x, y, p1, p2), PMX(y, x, p1, p2)


def crossover_populatie(pop: np.ndarray, valori: np.ndarray, cost: np.ndarray, pc: float) -> Tuple[np.ndarray, np.ndarray]:
    """Realizeaza recombinarea populatiei (PMX sau asexuata)."""
    dim, n = pop.shape
    po = np.zeros((dim, n), dtype = "int")
    val = np.zeros(dim, dtype="float")
    for i in range(0, dim - 1, 2):
        if np.random.uniform(0, 1) <= pc:
            c1, c2 = crossover_PMX(pop[i], pop[i + 1])
            v1, v2 = functie_fitness(cost, c1), functie_fitness(cost, c2)
        else:
            c1, c2 = pop[i].copy(), pop[i + 1].copy()
            v1, v2 = valori[i], valori[i + 1]
        po[i], po[i + 1] = c1, c2
        val[i], val[i + 1] = v1, v2
    return po, val

# =============================================================================
# 3. SELECȚIE (SUS & Elitism)
# =============================================================================
def fps(qual: np.ndarray, dim: int) -> np.ndarray:
    """Fitness Proportionate Selection - Distributie cumulata."""
    prob = qual / np.sum(qual)
    return np.cumsum(prob)

def sigmafps(qual: np.ndarray, dim: int) -> np.ndarray:
    """Scalare Sigma pentru a evita convergenta prematura."""
    med, var = np.mean(qual), np.std(qual)
    newq = np.array([max(0, qual[i] - (med - 2 * var)) for i in range(dim)])
    if np.sum(newq) == 0:
        return fps(qual, dim)
    return fps(newq, dim)

def SUS(pop: np.ndarray, qual: np.ndarray, dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Universal Sampling: Selectie cu variatie scazuta."""
    spop = pop.copy(dim)
    squal = np.zeros(dim)
    qfps = sigmafps(qual, dim)
    pas = 1.0 / dim
    r = np.random.uniform(0, pas)
    k, i = 0, 0
    while k < dim:
        while r <= qfps[i]:
            spop[k][:] = pop[i][:]
            squal[k] = qual[i]
            r += pas
            k += 1
        i += 1
    return spop, squal

def elitism(pop_c: np.ndarray, qual_c: np.ndarray, pop_mo: np.ndarray,
            qual_mo: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pastreaza cel mai bun individ din generatia parinte in generatia urmas."""
    pop_rez, qual_rez = np.copy(pop_mo), np.copy(qual_mo)
    if np.max(qual_c) > np.max(qual_mo):
        imax = np.argmax(qual_c)
        ir = np.random.randint(dim)
        pop_rez[ir] = pop_c[imax].copy()
        qual_rez[ir] = qual_c[imax]
    return pop_rez, qual_rez

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Parametri
    PM, PC, DIM = 0.2, 0.8, 30

    # 1. Generare (necesita fisierul costuri_TSP.txt)
    try:
        pop, val, cost_mat = genereaza_populatie("costuri_TSP.txt", DIM)
        print(f"Cel mai bun initial: {np.max(val)}")

        # 2. Variatie
        p_mut, v_mut = mutatie_populatie(pop, val, cost_mat, PM)
        p_cross, v_cross = crossover_populatie(pop, val, cost_mat, PC)

        # 3. Selectie
        parinti, v_parinti = SUS(pop, val, DIM, len(cost_mat))
        populatie_noua, v_nou = elitism(pop, val, p_mut, v_mut, DIM)

        print(f"Cel mai bun final (dupa o iteratie): {np.max(v_nou)}")
    except OSError:
        print("Fisierul 'costuri_TSP.txt' nu a fost gasit pentru test.")
