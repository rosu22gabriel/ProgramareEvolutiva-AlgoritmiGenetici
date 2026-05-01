import numpy as np
from typing import List, Tuple, Union, Set

# =============================================================================
# 1. CROSSOVER PENTRU VECTORI BINARI SAU ÎNTREGI
# =============================================================================
def crossover_unipunct(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover Unipunct: Taie parintii intr-un punct aleator si schimba segmentele.
    Optim pentru pastrarea structurii blocurilor de gene.
    """
    # i este punctul de taiere (intre 1 si n - 1 pentru a asigura schimbul de material)
    i = np.random.randint(1, n - 1)
    c1, c2 = x.copy(), y.copy()

    c1[i:n], c2[i:n] = y[i:n].copy(), x[i:n].copy()
    return c1, c2


def crossover_uniform(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover Uniform: Fiecare gena este preluata de la unul dintre parinti cu probabilitatea de 50%
    """
    c1, c2 = x.copy(), y.copy()
    for i in range(n):
        if np.random.randint(0, 2) == 1:
            c1[i], c2[i] = y[i], x[i]
    return c1, c2


# =============================================================================
# 2. CROSSOVER PENTRU PERMUTĂRI (Probleme tip TSP)
# =============================================================================
def crossover_PMX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Partially Mapped Crossover: Protejeaza un segment si mapeaza restul pentru a evita duplicatele"""
    poz = np.random.randint(0, n, 2)
    p1, p2 = np.min(poz), np.max(poz)
    while p1 == p2:
        poz = np.random.randint(0, n, 2)
        p1, p2 = np.min(poz), np.max(poz)

    return PMX(x, y, n, p1, p2), PMX(y, x, n, p1, p2)

def PMX(x: np.ndarray, y: np.ndarray, n: int, p1: int, p2: int) -> np.ndarray:
    """Functia suport pentru PMX."""
    c = -np.ones(n, dtype=int)
    c[p1:p2+1] = x[p1:p2+1]

    for i in range(p1, p2+1):
        a = y[i]
        if a not in c:
            curent = i
            plasat = False
            while not plasat:
                b = x[curent]
                # Unde se afla b in al doilea parinte?
                poz = np.where(y == b)[0][0]
                if c[poz] == -1:
                    c[poz] = a
                    plasat = True
                else:
                    curent = poz

    # Completarea pozitiilor ramase libere
    poz_libere = np.where(c == -1)[0]
    elemente_ramase = [el for el in y if el not in c]
    c[poz_libere] = elemente_ramase
    return c

def crossover_OCX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Order Crossover: Mentine ordinea relativa a oraselor din parintele secundar."""
    poz = np.random.randint(0, n, 2)
    p1, p2 = sorted(poz)
    if p1 == p2: return x.copy(), y.copy()

    return OCX(x, y, n, p1, p2), OCX(y, x, n, p1, p2)

def OCX(x: np.ndarray, y: np.ndarray, n: int, p1: int, p2: int) -> np.ndarray:
    """Functia suport pentru OCX."""
    copil = -np.ones(n, dtype=int)
    # 1. Copiem secventa selectata
    copil[p1:p2+1] = x[p1:p2+1]

    # 2. Colectam elementele din y incepand de la p2+1 in mod circular
    y_circular = np.concatenate((y[p2+1:], y[:p2+1]))
    elemente_y = [el for el in y_circular if el not in copil]

    # 3. Completam copilul circular incepand de la p2+1
    pozitii = list(range(p2 + 1, n)) + list(range(0, p1))
    for i, p in enumerate(pozitii):
        copil[p] = elemente_y[i]
    return copil


def cicluri(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Functie suport care identifica structura de cicluri intre doua permutari.
    Returneaza un vector unde fiecare pozitie indica indexul
    """

# =============================================================================
# 3. CROSSOVER PENTRU NUMERE REALE (Aritmetic)
# =============================================================================

def crossover_total(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover Aritmetic Total: Combina liniar toate genele.
    Copil = alpha * P1 + (1 - alpha) * P2
    """
    c1 = alpha * x + (1 - alpha) * y
    c2 = (1 - alpha) * x + alpha * y
    return c1, c2

def crossover_singular(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover Aritmetic Singular: Doar o gena aleatorie este mediata, restul sunt copiate.
    """
    i = np.random.randint(0, n)
    c1, c2 = x.copy(), y.copy()
    c1[i] = alpha * x[i] + (1 - alpha) * y[i]
    c2[i] = (1 - alpha) * x[i] + alpha * y[i]
    return c1, c2