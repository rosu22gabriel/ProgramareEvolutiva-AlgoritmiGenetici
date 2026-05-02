import numpy as np
import matplotlib.pyplot as grafic
import random
import copy
from typing import List, Tuple, Union

# =============================================================================
# 1. EVALUARE ȘI INIȚIALIZARE (Reprezentare Reală)
# =============================================================================
def calculeaza_fitness(cost: np.ndarray, valoare: np.ndarray, Cmax: float, x: List[float]) -> Tuple[bool, float]:
    """
    Evalueaza un individ (solutie candidat).

    Args:
        cost: Vectorul costurilor obiectelor.
        valoare: Vectorul valorile obiectelor.
        Cmax: Capacitatea maxima a rucsacului.
        x: Genotipul (proportia selectata din fiecare obiect, x_i in [0, 1]).

    Returns:
        (fezabil, valoare_totala): Daca solutia respecta restrictia de capacitate si
        profitul ei.
    """
    cost_x = sum([x[i] * cost[i] for i in range(len(cost))])
    valoare_x = sum([x[i] * valoare[i] for i in range(len(valoare))])
    # Solutia este fezabila doar daca nu depaseste Cmax
    return cost_x <= Cmax, valoare_x

def genereaza_populatie(cost: np.ndarray, valoare: np.ndarray, Cmax: float, dim: int) -> List[List[float]]:
    """
    Genereaza populatia initiala asigurandu-se ca toti indivizii sunt fezabili (respecta Cmax).
    Fiecare individ are structura: [x1, x2, ..., xn, fitness].
    """
    n = np.size(cost)
    pop = []
    for i in range(dim):
        gasit = False
        while not gasit:
            # Generam gene reale in intervalul [0, 1]
            x = [random.uniform(0, 1) for _ in range(n)]
            gasit, val_x = calculeaza_fitness(cost, valoare, Cmax, x)
        # Adaugam valoarea fitness la sfarsitul vectorului
        x_complet = x + [val_x]
        pop.append(x_complet)
    return pop

# =============================================================================
# 2. OPERATORI DE VARIAȚIE (Crossover Singular & Mutație Gaussiană)
# =============================================================================
def crossover_singular(x: List[float], y: List[float], n: int, alpha: float) -> Tuple[List[float], List[float]]:
    """
    Crossover Singular: Alege o gena aleatorie si aplica o combinatie convexa intre parinti.
    c1[i] = alpha * x[i] + (1 - alpha) * y[i]
    """
    c1, c2 = x.copy(), y.copy()
    i = random.randint(0, n - 1)
    c1[i] = alpha * x[i] + (1 - alpha) * y[i]
    c2[i] = alpha * y[i] + (1 - alpha) * x[i]
    return c1, c2

def crossover_populatie(pop,cost,valoare,Cmax,pc, alpha):
    dim=len(pop)
    n=cost.size
    po=copy.deepcopy(pop)
    #populatia este parcursa astfel incat sunt selectati indivizii 0,1 apoi 2,3 s.a.m.d
    for i in range(0,dim-1,2):
        #selecteaza parintii
        x = pop[i][:-1].copy()
        y = pop[i+1][:-1].copy()
        r = np.random.uniform(0,1)
        if r<=pc:
            # crossover x cu y - uniform: mai potrivit aici
            c1,c2=crossover_singular(x ,y ,n,alpha)
            fez, val = calculeaza_fitness(cost,valoare,Cmax, c1)
            if fez:
                po[i][:-1]=c1.copy()
                po[i][-1]=val
            fez, val = calculeaza_fitness(cost,valoare,Cmax, c2)
            if fez:
               po[i+1][:-1]=c2.copy()
               po[i+1][-1]=val
    return po

def mutatie_populatie(pop: List[List[float]], cost: np.ndarray, valoare: np.ndarray,
                      Cmax: float, pm: float, sigma: float) -> List[List[float]]:
    """
    Mutatie Neuniforma (Gaussiana): Adauga zgomot normal genelor cu probabilitatea pm.
    Daca mutatia produce un individ nefezabil, se pastreaza forma originala.
    """
    pop_m = copy.deepcopy(pop)
    dim = len(pop)
    n = cost.size

    for i in range(dim):
        x = pop[i][:n].copy()
        mutat = False
        for j in range(n):
            if random.uniform(0, 1) <= pm:
                # Adaugam zgomot Gaussian (media 0, deviatia sigma)
                p = np.random.normal(0, sigma)
                y = x[j] + p
                # Mentinem gena in intervalul valid [0, 1]
                x[j] = max(0, min(y, 1))
                mutat = True

        if mutat:
            fez, val = calculeaza_fitness(cost, valoare, Cmax, x)
            if fez:
                # Daca e valid, actualizam individul si fitness-ul
                pop_m[i] = x + [val]
    return pop_m

# =============================================================================
# 3. OPERATORI DE SELECȚIE (Turneu & Elitism)
# =============================================================================

def turneu(pop: np.ndarray, qual: np.ndarray, dim: int, n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selectia Turneu: Alege k indivizi aleatoriu si il pastreaza pe cel mai bun.
    """
    spop = np.zeros((dim, n))
    squal = np.zeros(dim)
    for it in range(dim):
        # Alegem indicii participantilor la turneu
        poz = np.random.randint(0, len(pop), k)
        v = [qual[p] for p in poz]
        # Identificam castigatorul
        imax = np.argmax(v)
        isel = poz[imax]

        spop[it][:] = pop[isel][:]
        squal[it] = qual[isel]
    return spop, squal

def elitism(pop_c: np.ndarray, qual_c: np.ndarray, pop_mo: np.ndarray,
            qual_mo: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Asigura supravietuirea celui mai bun individ din generatia precedenta (pop_c)
    in noua generati (pop_mo).
    """
    pop_rez = np.copy(pop_mo)
    qual_rez = np.copy(qual_mo)

    if np.max(qual_c) > np.max(qual_mo):
        imax = np.argmax(qual_c)
        # Inlocuim un individ aleatoriu din noua generatie cu elitistul
        ir = np.random.randint(dim)
        pop_rez[ir] = pop_c[imax].copy()
        qual_rez[ir] = qual_c[imax]
    return pop_rez, qual_rez

# =============================================================================
# FLUXUL PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    # Parametri problemei
    C_MAX = 50
    DIM_POP = 10
    PM = 0.2
    PC = 0.8
    ALPHA = 0.3
    SIGMA = 0.7 # Intensitatea mutatiei

    # Incarcare date (presupunem fisierele existente)
    # cost = np.genfromtxt("cost_cont.txt")
    # valoare = np.genfromtxt("valoare_cont.txt")
    cost = np.array([10, 20, 30, 40, 50])
    valoare = np.array([15, 25, 35, 45, 55])

    # 1. Generare
    populatie = genereaza_populatie(cost, valoare, C_MAX, DIM_POP)

    # 2. Variație
    pop_recombinata = crossover_populatie(populatie, cost, valoare, C_MAX, PC, ALPHA)
    pop_mutata = mutatie_populatie(pop_recombinata, cost, valoare, C_MAX, PM, SIGMA)

    # 3. Pregătire date pentru selecție (separare fitness)
    pop_genotipuri = np.array([ind[:-1] for ind in populatie])
    pop_fitness = np.array([ind[-1] for ind in populatie])

    mut_genotipuri = np.array([ind[:-1] for ind in pop_mutata])
    mut_fitness = np.array([ind[-1] for ind in pop_mutata])

    # 4. Selecție
    parinti, fitness_p = turneu(pop_genotipuri, pop_fitness, DIM_POP, cost.size, 2)
    generatia_urmatoare, fitness_g = elitism(pop_genotipuri, pop_fitness, mut_genotipuri, mut_fitness, DIM_POP)

    print(f"Cel mai bun fitness găsit: {np.max(fitness_g)}")
