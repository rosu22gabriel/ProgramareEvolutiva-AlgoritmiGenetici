# Selectia este mecanismul care decide care indivizi  vor supravietui sau vor deveni parinti
# bazandu-se pe principiul "supravietuirii celui mai adaptat" (survival of the fittest)

import numpy as np
from typing import Tuple, List, Union

# =============================================================================
# 1. SELECȚIA TURNEU (Tournament Selection)
# =============================================================================
def turneu(pop: np.ndarray, qual: np.ndarray, dim: int, n: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selectia Turneu: Alege k indivizi aleatoriu si il selecteaza pe cel mai bun.
    Repeta procesul de 'dim' ori pentru a forma noua populatie.

    Args:
        k: Dimensiunea turneului (presiunea de selectie creste odata cu k).
    """
    spop = pop.copy()
    squal = np.zeros(dim)
    for it in range(dim):
        # Alegem k pozitii aleatorii din populatie
        poz = np.random.randint(0, dim, k)
        # Identificam cea mai buna calitate din sub-grupul ales
        v = [qual[poz[i]] for i in range(k)]
        imax = np.argmax(v)
        isel = poz[imax]

        spop[it][:] = pop[isel][:]
        squal[it] = qual[isel]
    return spop, squal


# =============================================================================
# 2. SELECȚIA PROPORȚIONALĂ (Roulette Wheel / SUS)
# =============================================================================

def sigmafps(qual: np.ndarray, dim: int) -> np.ndarray:
    """
    Sigma-Scalare: Ajusteaza calitatile pentru a evita dominanta unui individ foarte bun
    (super-individ) sau stagnarea cand calitatile sunt foarte apropiate.
    """
    med = np.mean(qual)
    var = np.std(qual)
    # Calculeaza calitatea ajustata: max(0, f(x) - (media - 2*sigma))
    newq = np.array([max(0, qual[i] - (med - 2 * var)) for i in range(dim)])

    if np.sum(newq) == 0:
        return fps(qual, dim)
    return fps(newq, dim)

def fps(qual: np.ndarray, dim: int) -> np.ndarray:
    "Fitness Proportionate Selection (FPS): Calculeaza probabilitatile cumulate."
    suma = np.sum(qual)
    prob = qual/suma
    return np.cumsum(prob)

def ruleta(pop: np.ndarray, qual: np.ndarray, dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selectia Ruleta: Probabilitatea de selectie este proportionala cu calitatea.
    Foloseste sigma-scalare pentru o distributie mai echilibrata.
    """
    spop = pop.copy()
    squal = np.zeros(dim)
    qfps = sigmafps(qual, dim)
    for it in range(dim):
        r = np. random.uniform(0, 1)
        isel = np.where(qfps >= r)[0][0]
        spop[it][:] = pop[isel][:]
        squal[it] = qual[isel]
    return spop, squal

def SUS(pop: np.ndarray, qual: np.ndarray, dim: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stochastic Universal Sampling (SUS): O varianta de ruleta cu 'ace' echidistante.
    Reduce varianta selectiei si este mai stabila decat ruleta simpla.
    """
    spop = pop.copy()
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


# =============================================================================
# 3. SELECȚIA BAZATĂ PE RANG (Rank Selection)
# =============================================================================
def lrang(dim: int, s: float) -> np.ndarray:
    """
    Rang Liniar: Probabilitatea depinde de pozitia in clasament, nu de valoarea fitness-ului.
    Elimina problemele de scalare a functiei obiectiv.
    """
    # s: presiunea de selectie, de obicei in [1.0, 2.0]
    lr = [(2 - s) / dim + 2 * (i + 1) * (s - 1) / (dim * (dim + 1)) for i in range(dim)]
    return np.cumsum(lr)

# =============================================================================
# 4. SELECȚIA PENTRU NOUA GENERAȚIE (Elitism / Genitor / Deterministă)
# =============================================================================
def elitism(pop_c: np.ndarray, qual_c: np.ndarray, pop_mo: np.ndarray, qual_mo: np.ndarray, dim: Int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Elitism: Se asigura ca cel mai bun individ din generatia anterioara.
    este pastrat in noua generatie (inlocuieste un individ aleatoriu daca e nevoie).
    """
    pop, qual = np.copy(pop_mo), np.copy(qual_mo)
    max_c, max_mo = np.max(qual_c), np.max(qual_mo)

    if max_c > max_mo:
        imax = np.argmax(qual_c)
        ir = np.random.randint(dim)
        pop[ir] = pop_c[imax].copy()
        qual[ir] = max_c
    return pop, qual

def sel_det(pop_c: np.ndarray, qual_c: np.ndarray, pop_mo: np.ndarray, qual_mo: np.ndarray, dim: int, L: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Selectia determinista (u + l): Uneste parintii cu copiii si ii alege pe cei mai buni
    'dim' indivizi din total de 2 * dim.
    """
    pop_total = np.vstack((pop_c, pop_mo))
    qual_total = np.concatenate((qual_c, qual_mo))

    # Sortam descrescator dupa calitate si luam primii 'dim'
    indici = np.argsort(qual_total)[-dim:]
    pop_r = pop_total[indici]
    qual_r = qual_total[indici]

    # Amestecam rezultatul pentru a evita ordonarea in populatie
    p = np.random.permutation(dim)
    return pop_r[p], qual_r[p]













