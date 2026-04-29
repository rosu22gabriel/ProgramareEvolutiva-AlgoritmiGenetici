import copy
import numpy as np
import random
import matplotlib.pyplot as grafic
from typing import List, Tuple, Union

# --- Enunt Cerinta ---
# Problema Comis-Voiajorului (TSP):
# Fiind data o lista de orase si distantele dintre ele (matricea de costuri),
# trebuie gasit cel mai scurt drum care viziteaza fiecare oras exact o data
# si se intoarce la orasul de plecare.
# Reprezentare: Permutare a indicilor oraselor.
# ---------------------

def functie_fitness(cost: np.ndarray, p: np.ndarray) -> float:
    """
    Calculeaza fitness-ul ca fiind 100 / Cost_Total.
    Un drum mai scurt va rezulta intr-o valoare de fitness mai mare.
    """
    cost_p = 0
    n = len(p)
    for i in range(n - 1):
        # Adunam costul de la orasul curent la urmatorul
        cost_p += cost[p[i], p[i+1]]
    # Adunam costul de la ultimul oras inapoi la primul (inchidem circuitul)
    cost_p += cost[p[0], p[n-1]]
    return 100 / cost_p

def mutatie_inversiune(p: np.ndarray) -> np.ndarray:
    """
    Operator de Mutatie prin Inversiune:
    Alege doua puncte aleatorii si inverseaza ordinea genelor intre ele.
    Este mult mai eficienta pentru TSP decat interschimbarea simpla deoarece
    pastreaza majoritatea legaturilor (adiacentelor) dintre orase.
    """
    n = p.size
    i = np.random.randint(0, n - 1)
    j = np.random.randint(i + 1, n)
    r = p.copy()
    # Inversam secventa p[i...j]
    r[i:j+1] = [p[k] for k in range(j, i - 1, -1)]
    return r

def mutatie_populatie(pop: np.ndarray, val_pop: np.ndarray, cost: np.ndarray, pm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica mutatia prin inversiune la nivelul intregii populatii.
    """
    pop_m = pop.copy()
    val_pop_m = val_pop.copy()
    dim = pop_m.shape[0]

    for i in range(dim):
        if np.random.uniform(0, 1) <= pm:
            x = pop_m[i]
            y = mutatie_inversiune(x)
            pop_m[i] = y
            val_pop_m[i] = functie_fitness(cost, y)
    return pop_m, val_pop_m

def PMX(x: np.ndarray, y: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """
    Partially Mapped Crossover (PMX) - Creeaza un copil din doi parinti.
    Se asigura ca nu exista duplicate in permutare prin mapari succesive.
    """
    n = x.size
    copil = - np.ones(n, dtype="int")
    # Pas 1: Copiem secventa centrala din primul parinte
    copil[p1: p2+1] = x[p1:p2 + 1]

    # Pas 2: Mapam elementele din y care cad in zona p1:p2
    poz_alele_neplasate = [k for k in range(p1, p2 + 1) if not y[k] in copil]
    for p in poz_alele_neplasate:
        valoare_de_plasat = y[p]
        # Cautam pozitia libera prin mapare
        while copil[p] > -1:
            valoare_conflict = copil[p]
            p = np.where(y == valoare_conflict)[0][0]
        copil[p] = valoare_de_plasat

    # Pas 3: Completam restul pozitiilor cu restul elementelor din y
    alele_neplasate = [y[k] for k in range(n) if not y[k] in copil]
    pozitii_libere = [k for k in range(n) if copil[k] == -1]
    copil[pozitii_libere] = alele_neplasate
    return copil

def crossover_populatie(pop: np.ndarray, valori: np.ndarray, cost: np.ndarray, pc: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realizeaza recombinarea populatiei folosind operatorul PMX.
    """
    dim, n = pop.shape
    po = np.zeros((dim, n), type="int")
    val = np.zeros(dim, dtype="float")

    for i in range(0, dim - 1, 2):
        x = pop[i]
        y = pop[i + 1]

        if np.random.uniform(0, 1) <= pc:
            # Puncte de taiere aleatorii pentru PMX
            p1 = np.random.randint(0, n - 1)
            p2 = np.random.randint(p1 + 1, n)

            c1 = PMX(x, y, p1, p2)
            c2 = PMX(y, x, p1, p2)
            v1 = functie_fitness(cost, c1)
            v2 = functie_fitness(cost, c2)

        else:
            # Recombinarea asexuata (copiem parintii)
            c1, c2 = x.copy(), y.copy()
            v1, v2 = valori[i], valori[i + 1]

        po[i], po[i+1] = c1, c2
        val[i], val[i+1] = v1, v2

    return po, val

def genereaza_populatie(file_path: str, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Citeste matricea de costuri si genereaza o populatie initiala aleatorie.
    """
    try:
        cost = np.genfromtxt(file_path)
    except:
        # Fallback daca fisierul lipseste
        print("Fisier costuri negasit. Generam matrice aleatorie...")
        cost = np.random.randint(10, 100, (10, 10))
        np.fill_diagonal(cost, 0)

    n = len(cost)
    pop = np.array([np.random.permutation(n) for _ in range(dim)])
    val_pop = np.array([functie_fitness(cost, ind) for ind in pop])
    return pop, val_pop, cost

if __name__ == "__main__":
    # Parametri algoritmului
    PM = 0.2
    PC = 0.8
    DIM_POP = 30

    # Rulare simulare pe un singur pas
    p, v, matrice_cost = genereaza_populatie("costuri_TSP.txt", DIM_POP)

    idx_best_init = np.argmax(v)
    print(f"Cel mai bun individ initial : {p[idx_best_init]}, Fitness: {v[idx_best_init]:.4f}")

    pop_m, val_m = mutatie_populatie(p, v, matrice_cost, PM)
    print(f"Cel mai bun dupa mutatie    : {pop_m[np.argmax(val_m)]}, Fitness: {np.max(val_m):.4f}")

    pop_c, val_c = crossover_populatie(p, v, matrice_cost, PC)
    print(f"Cel mai bun dupa crossover:  {pop_c[np.argmax(val_c)]},  Fitness: {np.max(val_c):.4f}")
