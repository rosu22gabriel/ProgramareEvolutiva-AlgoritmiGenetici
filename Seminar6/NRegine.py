import numpy as np
import matplotlib.pyplot as grafic
import random
from typing import List, Tuple, Union


# --- ENUNȚ CERINȚĂ ---
# Problema celor N Regine: Plasați N regine pe o tablă de șah de N x N astfel încât
# să nu existe două regine care să se atace reciproc (pe rânduri, coloane sau diagonale).
#
# Reprezentare: Permutare de n elemente (evită automat atacul pe rânduri și coloane).
# Funcția obiectiv (Maxim): Numărul maxim de perechi de regine care NU se atacă pe diagonale.
# Pentru n regine, numărul total de perechi este n*(n-1)/2. O soluție are acest fitness.
# ---------------------

def calculeaza_functie_obiectiv(n: int, x: np.ndarray) -> float:
    """
    Calculează fitness-ul unui individ.
    Pornim de la numărul maxim de perechi posibile și scădem conflictele pe diagonală.

    Args:
        n: Numărul de regine (dimensiunea tablei).
        x: Permutarea reprezentând pozițiile reginelor.
    """
    c = n * (n - 1) / 2
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Două regine se atacă pe diagonală dacă distanța dintre coloane
            # este egală cu distanța dintre rânduri.
            if abs(i - j) == abs(x[i] - x[j]):
                c -= 1
    return c


def mutatie_inserare(p: np.ndarray, n: int) -> np.ndarray:
    """
    Operator de mutație prin inserare: extrage o genă și o inserează pe o altă poziție.

    Args:
        p: Permutarea originală.
        n: Lungimea permutării.
    """
    p1 = np.random.randint(0, n - 1)
    p2 = np.random.randint(p1 + 1, n)
    rez = p.copy()
    if p1 + 1 < p2:
        el = p[p2]
        rez = np.delete(rez, p2)
        rez = np.insert(rez, p1 + 1, el)
    return rez


def mutatie_populatie(pop: np.ndarray, dim: int, n: int, pm: float) -> np.ndarray:
    """
    Aplică mutația de inserare la nivelul întregii populații cu probabilitatea pm.
    """
    pop_m = pop.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        if r <= pm:
            # Extragem genele (fără fitness), mutăm și actualizăm fitness-ul
            x = mutatie_inserare(pop_m[i, :n], n)
            pop_m[i, :n] = x.copy()
            pop_m[i, n] = calculeaza_functie_obiectiv(n, x)
    return pop_m


def crossover_CX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Operatorul Cycle Crossover (CX): realizează recombinarea bazată pe cicluri
    pentru a conserva pozițiile valorilor din părinți.
    """
    ciclu = cicluri(x, y, n)
    c1 = x.copy()
    c2 = y.copy()
    for i in range(n):
        # Se folosește divmod pentru a verifica dacă indexul ciclului este par sau impar
        cat, rest = np.divmod(ciclu[i], 2)
        # Interschimbăm alelele din ciclurile pare pentru a crea diversitate
        if not rest:
            c1[i] = y[i]
            c2[i] = x[i]
    return c1, c2


def cicluri(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Identifică ciclurile formate între doi părinți (folosit pentru crossover-ul CX).

    Returns:
        Un vector unde fiecare poziție conține indexul ciclului din care face parte gena.
    """
    index = 1
    ciclu = np.zeros(n)
    gata = 0
    while not gata:
        p = np.where(ciclu == 0)
        if np.size(p):
            i = p[0][0]
            a = x[i]
            ciclu[i] = index
            b = y[i]
            while b != a:
                # Căutăm valoarea b în părintele x pentru a închide ciclul
                r = np.where(x == b)
                j = r[0][0]
                ciclu[j] = index
                b = y[j]
            index += 1
        else:
            gata = 1
    return ciclu


def crossover_populatie(pop: np.ndarray, pc: float) -> np.ndarray:
    """
    Realizează recombinarea populației prin perechi consecutive (0-1, 2-3...).
    """
    dim = pop.shape[0]
    m = pop.shape[1]
    n = m - 1
    po = pop.copy()
    for i in range(0, dim - 1, 2):
        x = pop[i, :-1]
        y = pop[i + 1, :-1]
        r = np.random.uniform(0, 1)
        if r <= pc:
            # Recombinăm părinții folosind CX
            c1, c2 = crossover_CX(x, y, n)
            # Evaluăm noii copii
            val1 = calculeaza_functie_obiectiv(n, c1)
            val2 = calculeaza_functie_obiectiv(n, c2)
            po[i][:n] = c1.copy()
            po[i][n] = val1
            po[i + 1][:n] = c2.copy()
            po[i + 1][n] = val2
    return po


def genereaza_populatie(n: int, dim: int) -> np.ndarray:
    """
    Generează o populație inițială de dimensiune dim.
    """
    pop = np.zeros([dim, n + 1], dtype=int)
    for i in range(dim):
        pop[i, :n] = np.random.permutation(n)
        pop[i, n] = calculeaza_functie_obiectiv(n, pop[i, :n])
    return pop


if __name__ == "__main__":
    # Parametrii algoritmului
    N_REGINE = 8
    DIM_POP = 30
    PROB_MUTATIE = 0.25
    PROB_CROSSOVER = 0.8

    # 1. Inițializare
    populatie = genereaza_populatie(N_REGINE, DIM_POP)
    print(f"Cel mai bun individ inițial : {populatie[np.argmax(populatie[:, -1]), :N_REGINE]}, "
          f"calitate {np.max(populatie[:, -1])}")

    # 2. Testare operator Mutație
    pop_m = mutatie_populatie(populatie, DIM_POP, N_REGINE, PROB_MUTATIE)
    print(f"Cel mai bun individ după mutație : {pop_m[np.argmax(pop_m[:, -1]), :N_REGINE]}, "
          f"calitate {np.max(pop_m[:, -1])}")

    # 3. Testare operator Crossover
    pop_c = crossover_populatie(populatie, PROB_CROSSOVER)
    print(f"Cel mai bun individ după crossover : {pop_c[np.argmax(pop_c[:, -1]), :N_REGINE]}, "
          f"calitate {np.max(pop_c[:, -1])}")