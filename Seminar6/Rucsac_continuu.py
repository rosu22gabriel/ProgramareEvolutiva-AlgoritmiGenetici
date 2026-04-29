import numpy as np
import matplotlib.pyplot as grafic
import random
import copy
from typing import List, Tuple, Union

# --- Enunt Cerinta ---
# Problema Rucsacului Continua:
# Avem n obiecte, fiecare cu un cost si o valoare. Trebuie sa determinam
# fractiunile x[i] (0 <= x[i] <= 1) din fiecare obiect astfel incat
# suma costurilor sa nu depaseasca Cmax, iar valoarea totala sa fie maxima.
# ---------------------

def calculeaza_fitness(cost: np.ndarray, valoare: np.ndarray, Cmax: float, x: List[float]) -> Tuple[bool, float]:
    """
    Verifica fezabilitatea solutiei si calculeaza valoarea totala.
    Args:
        cost: Vectorul costurilor obiectelor.
        valoare: Vectorul valorilor obiectelor.
        Cmax: Capacitatea maxima a rucsacului.
        x: Genotipul (fractiunile alese din fiecare obiect).
    Returns:
        (este_fezabil, valoare_totala)
    """
    cost_x = sum([x[i] * cost[i] for i in range(len(cost))])
    valoare_x = sum([x[i] * valoare[i] for i in range(len(valoare))])
    # O solutie este fezabila daca nu depaseste Cmax.
    return cost_x <= Cmax, valoare_x


def mutatie_populatie(pop: List[List[float]], cost: np.ndarray, valoare: np.ndarray, Cmax: float, pm: float, sigma: float) -> List[List[float]]:
    """
    Aplica mutatia neuniforma (Gaussiana) asupra populatiei.

    Args:
        pm: Probabilitatea de mutatie per gena.
        sigma: Parametrul de dispersie pentru perturbare.
    """
    pop_m = copy.deepcopy(pop)
    dim = len(pop)
    n = cost.size

    for i in range(dim):
        # x este individul curent (fara coloana de fitness)
        x = pop[i][:n].copy()
        mutat = False
        for j in range(n):
            if random.uniform(0, 1) <= pm:
                # Generam o perturbare normala N(0, sigma)
                p = np.random.normal(0, sigma)
                y = x[j] + p
                # Clamping: fortam valoarea in intervalul [0, 1]
                if y > 1: y = 1
                if y < 0: y = 0
                x[j] = y
                mutat = True

        if mutat:
            fez, val = calculeaza_fitness(cost, valoare, Cmax, x)
            if fez:
                # Daca noua solutie este valida, o salvam (cu fitness-ul nou la final)
                pop_m[i] = x + [val]
    return pop_m


def genereaza_populatie(cost: np.ndarray, valoare: np.ndarray, Cmax: float, dim: int) -> List[List[float]]:
    """
    Genereaza o populatie initiala de indivizi cu valori reale aleatorii.
    Se asigura ca fiecare individ generat este fezabil (respecta Cmax).
    """
    n = np.size(cost)
    pop = []
    for i in range(dim):
        gasit = False
        while not gasit:
            # Fiecare gena este o fractiune intre 0 si 1
            x = [random.uniform(0, 1) for _ in range(n)]
            gasit, val_x = calculeaza_fitness(cost, valoare, Cmax, x)
        pop.append(x + [val_x])
    return pop

def crossover_singular(x: List[float], y: List[float], n: int, alpha: float) -> Tuple[List[float], List[float]]:
    """
    Crossover Aritmetic Singular: combina liniar doar o singura gena alaeasa aleator.

    Args:
        alpha: Factor de pondere (0 < alpha < 1)
    """
    c1 = x.copy()
    c2 = y.copy()
    i = random.randint(0, n - 1)

    # Noua valoare este o medie ponderata a valorilor parintilor
    c1[i] = alpha * x[i] + (1 - alpha) * y[i]
    c2[i] = alpha * y[i] + (1 - alpha) * x[i]

    return c1, c2

def crossover_populatie(pop: List[List[float]], cost: np.ndarray, valoare: np.ndarray,
                        Cmax: float, pc: float, alpha: float) -> List[List[float]]:
    """
    Realizeaza recombinarea populatiei folosind crossover aritmetic singular.
    Daca copiii rezultati nu sunt fezabili, parintii sunt pastrati.
    """
    dim = len(pop)
    n = cost.size
    po = copy.deepcopy(pop)

    for i in range(0, dim - 1, 2):
        # Selectam parintii consecutivi
        parinte1 = pop[i][:-1].copy()
        parinte2 = pop[i+1][:-1].copy()

        if random.uniform(0, 1) <= pc:
            c1, c2 = crossover_singular(parinte1, parinte2, n, alpha)

            # Verificam copilul 1
            fez1, val1 = calculeaza_fitness(cost, valoare, Cmax, c1)
            if fez1:
                po[i] = c1 + [val1]
            # Verificam copilul 2
            fez2, val2 = calculeaza_fitness(cost, valoare, Cmax, c2)
            if fez2:
                po[i + 1] = c2 + [val2]
    return po

if __name__ == "__main__":
    # Configurare
    CAPACITATE_MAXIMA = 50
    DIM_POPULATIE = 10
    PROB_MUTATIE = 0.2
    PROB_CROSSOVER = 0.8
    ALPHA = 0.3
    SIGMA = 0.7

    # Citire date
    # cost = np.random.uniform(5, 20, 10)
    # valoare = np.random.uniform(10, 50, 10)
    try:
        cost = np.genfromtxt("cost_cont_txt")
        valoare = np.genfromtxt("valoare_cont.txt")
    except:
        print("Fisierele nu au fost gasite. Se folosesc date simulate.")
        cost = np.array([10, 20, 30, 15, 25])
        valoare = np.array([100, 200, 150, 180, 220])

    # Rulare pasi algoritm
    pop_init = genereaza_populatie(cost, valoare, CAPACITATE_MAXIMA, DIM_POPULATIE)
    print(f"Cel mai bun Initial: {max(pop_init, key=lambda x: x[-1])[-1]:.2f}")

    pop_m = mutatie_populatie(pop_init, cost, valoare, CAPACITATE_MAXIMA, PROB_MUTATIE, SIGMA)
    print(f"Cel mai bun Mutatie: {max(pop_m, key=lambda x: x[-1])[-1]:.2f}")

    pop_c = crossover_populatie(pop_init, cost, valoare, CAPACITATE_MAXIMA, PROB_CROSSOVER, ALPHA)
    print(f"Cel mai bun Crossover: {max(pop_c, key=lambda x: x[-1])[-1]:.2f}")