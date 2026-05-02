import numpy as np
import matplotlib.pyplot as grafic
from typing import List, Tuple, Union

# =============================================================================
# 1. EVALUARE ȘI INIȚIALIZARE
# =============================================================================

def calculeaza_functie_obiectiv(n: int, x: np.ndarray) -> float:
    """
    Calculeaza calitate (fitness) pentru problema celor N-Regine.
    Valoarea maxima (fara conflicte) este n*(n-1)/2.
    Se scade cate o unitate pentru fiecare pereche de regine aflata pe aceeasi diagonala.
    """
    c = n * (n - 1) / 2
    for i in range(n - 1):
        for j in range(i + 1, n):
            # In reprezentarea prin permutari, verificam doar conflictele pe diagonala
            if abs(i - j) == abs(x[i] - x[j]):
                c -= 1
    return float(c)

def genereaza_populatie(n: int, dim: int) -> np.ndarray:
    """
    Genereaza o populatie initiala de permutari.
    Fiecare rand are n elemente (pozitiile reginelor) + 1 element (valoare fitness).
    """
    pop = np.zeros([dim, n + 1], dtype=int)
    for i in range(dim):
        pop[i, :n] = np.random.permutation(n)
        pop[i, n] = calculeaza_functie_obiectiv(n, pop[i, :n])
    return pop

# =============================================================================
# 2. OPERATORI DE MUTAȚIE (Includere/Inserare)
# =============================================================================

def mutatie_inserare(p: np.ndarray, n: int) -> np.ndarray:
    """
    Realizeaza mutatia prin inserare: extrage elementul de la pozitia p2 si il insereaza la pozitia p1 + 1.
    """
    p1 = np.random.randint(0, n - 1)
    p2 = np.random.randint(p1 + 1, n)
    rez = p.copy()

    if p1 + 1 < p2:
        el = p[p2]
        rez = np.delete(rez, p2)
        rez = np.insert(rez, p1 + 1, el)
    return rez

def mutatie_populatie(pop: np.ndarray, dim: int, n : int, pm: float) -> np.ndarray:
    """
    Aplica operatorul de mutatie asupra intregii populatii cu probabilitatea pm.
    """
    pop_m = pop.copy()
    for i in range(dim):
        r = np.random.uniform(0, 1)
        if r <= pm:
            # Extragem doar genotipul (primele n elemente)
            x = mutatie_inserare(pop_m[i, :n], n)
            pop_m[i, :n] = x
            # Recalculam fitness-ul pentru noul individ
            pop_m[i, n] = calculeaza_functie_obiectiv(n, x)
    return pop_m

# =============================================================================
# 3. OPERATORI DE RECOMBINARE (Cycle Crossover - CX)
# =============================================================================

def cicluri(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Identifica structurile de tip ciclu intre doi parinti (necesar pentru CX.)"""
    index = 1
    ciclu = np.zeros(n, dtype=int)
    gata = 0
    while not gata:
        p = np.where(ciclu == 0)[0]
        if len(p) > 0:
            i = p[0]
            val_start = x[i]
            ciclu[i] = index
            val_urmatoare = y[i]
            while val_urmatoare != val_start:
                j = np.where(x == val_urmatoare)[0][0]
                ciclu[j] = index
                val_urmatoare = y[j]
            index += 1
        else:
            gata = 1
    return ciclu


def crossover_CX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Crossover de tip Ciclu: conserva pozitiile absolute ale genelor."""
    ciclu = cicluri(x, y, n)
    c1, c2 = x.copy(), y.copy()
    for i in range(n):
        # Interschimbam genele doar pentru ciclurile cu index par
        if ciclu[i] % 2 == 0:
            c1[i], c2[i] = y[i], x[i]
    return c1, c2

def crossover_populatie(pop: np.ndarray, pc: float) -> np.ndarray:
    """Realizeaza recombinarea populatiei intre perechi de parinti consecutivi."""
    dim, m = pop.shape
    n = m - 1
    po = pop.copy()
    for i in range(0, dim - 1, 2):
        if np.random.uniform(0, 1) <= pc:
            parinte1, parinte2 = pop[i, :n], pop[i + 1, :n]
            c1, c2 = crossover_CX(parinte1, parinte2, n)

            # Actualizam copiii si fitness-ul lor in noua populatie
            po[i, :n], po[i, n] = c1, calculeaza_functie_obiectiv(n, c1)
            po[i+1, :n], po[i+1, n] = c2, calculeaza_functie_obiectiv(n, c2)
    return po

# =============================================================================
# 4. MECANISME DE SELECȚIE
# =============================================================================

def sigmafps(qual: np.ndarray, dim: int) -> np.ndarray:
    """Scalare Sigma pentru a normaliza presiunea de selectie."""
    med = np.mean(qual)
    var = np.std(qual)
    # Evitam valorile negative de fitness
    newq = np.array([max(0, qual[i] - (med - 2 * var)) for i in range(dim)])

    if np.sum(newq) == 0:
        # Daca toti sunt la fel, aplicam FPS standard
        prob = np.ones(dim) / dim
    else:
        prob = newq / np.sum(newq)
    return np.cumsum(prob)

def ruleta(pop: np.ndarray, qual: np.ndarray, dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Selectie tip Ruleta bazat pe probabilitati cumulate."""
    spop = np.zeros((dim, n))
    squal = np.zeros(dim)
    qfps = sigmafps(qual, dim)

    for it in range(dim):
        r = np.random.uniform(0, 1)
        isel = np.where(qfps >= r)[0][0]
        spop[it, :] = pop[isel, :n]
        squal[it] = qual[isel]
    return spop, squal


def elitism(pop_p: np.ndarray, qual_p: np.ndarray, pop_c: np.ndarray, qual_c: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pastreaza cel mai bun individ din generatia anterioara in cea noua."""
    pop_nou, qual_nou = pop_c.copy(), qual_c.copy()

    if np.max(qual_p) > np.max(qual_c):
        best_idx = np.argmax(qual_p)
        replace_idx = np.random.randint(dim)
        pop_nou[replace_idx] = pop_p[best_idx]
        qual_nou[replace_idx] = qual_p[best_idx]
    return pop_nou, qual_nou

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == "__main__":
    N_REGINE = 8
    DIM_POPULATIE = 30
    PROB_MUTATIE = 0.25
    PROB_CROSSOVER = 0.8

    # Initializare
    populatie = genereaza_populatie(N_REGINE, DIM_POPULATIE)
    print(f"Fitness maxim initial: {np.max(populatie[:, -1])}")

    # 2. Recombinare
    pop_dupa_cross = crossover_populatie(populatie, PROB_CROSSOVER)

    # 3. Mutatie
    pop_dupa_mutatie = mutatie_populatie(pop_dupa_cross, DIM_POPULATIE, N_REGINE, PROB_MUTATIE)

    # 4. Selectie elitista pentru a forma generatia urmatoare
    gen_nou, fitness_nou = elitism(
        populatie[:, :N_REGINE], populatie[:, N_REGINE],
        pop_dupa_mutatie[:, :N_REGINE], pop_dupa_mutatie[:, N_REGINE],
        DIM_POPULATIE
    )

    print(f"Fitness maxim dupa o generatie: {np.max(fitness_nou)}")
