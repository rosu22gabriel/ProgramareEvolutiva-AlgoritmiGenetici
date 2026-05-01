import numpy as np
from typing import List, Union


# Mutatia are rolul de a introduce variabilitate genetica noua in populatie
# prevenind convergenta prematura catre un optim local.


# =============================================================================
# 1. MUTATIE PENTRU VECTORI BINARI
# =============================================================================
def m_binar(x: int) -> int:
    """
    Mutatia Bit-Flip: Inverseaza valoarea unui bit (0 devine 1 si invers).
    Este operatorul standard pentru reprezentarile binare.
    """
    y = not x
    return int(y)

# =============================================================================
# 2. MUTATIE PENTRU NUMERE ÎNTREGI
# =============================================================================
def m_ra(a: int, b: int) -> int:
    """
    Resetare Aleatoare: Inlocuieste valoarea curenta cu o valoare aleatorie
    din intervalul [a, b)
    """
    y = np.random.randint(a, b)
    return y

def m_fluaj(x: int, a: int, b: int) -> int:
    """
    Mutatia Fluaj (Creep Mutation): Modifica valoarea curent cu o unitate (+1 sau -1),
    asigurandu-se ca rezulatul ramane in limitele [a, b].
    Este utila pentru explorarea locala fina in reprezentari intregi.
    """
    sign = np.random.choice([-1, 1])
    y = x + sign

    # Mentinerea in intervalul [a, b]
    return max(a, min(y, b))

# =============================================================================
# 3. MUTATIE PENTRU NUMERE REALE
# =============================================================================
def m_uniforma(a: float, b: float) -> float:
    """
    Mutatia Uniforma: Generare unei valori noi in intervalul [a, b]
    urmand o distributie uniforma.
    """
    return float(np.random.uniform(a, b))


def m_neuniforma(x: float, sigma: float, a: float, b: float) -> float:
    """
    Mutatia Neuniforma (Gaussiana): Adauga un zgomot repartizat normal (Gauss)
    cu media 0 si deviatia standard sigma peste valoarea curenta.

    Args:
        x: Valoarea genei.
        sigma: Pasul Mutatiei (deviatia standard).
        a, b: Limitele domeniului de definitie.
    """
    p = np.random.normal(0, sigma)
    y = x + p


    # Verificarea limitelor intervalului
    return float(max(a, min(y, b)))


# =============================================================================
# 4. MUTATIE PENTRU PERMUTĂRI (TSP / Programare)
# =============================================================================
def m_perm_inversiune(x: np.ndarray, n: int) -> np.ndarray:
    """
    Mutatia prin Inversiune:
    Selecteaza doua puncte si inverseaza ordinea elementelor intre ele.
    Conserva adiacenta genelor.
    """
    poz = np.random.choice(range(n), 2, replace="False")
    p1, p2 = sorted(poz)

    y = x.copy()
    y[p1:p2+1] = x[p1:p2+1][::-1]
    return y


def m_interschimbare(x: np.ndarray, n: int) -> np.ndarray:
    """
    Mutatia prin Interschimbare (Swap): Schimba pozitia a doua gene alese aleatoriu.
    """
    idx1, idx2 = np.random.choice(range(n), 2, replace=False)
    y = x.copy()
    y[idx1], y[idx2] = x[idx2], x[idx1]
    return y


def m_perm_inserare(x: np.ndarray, n: int) -> np.ndarray:
    """
    Mutatia prin Inserare: Extrage o gena de la o pozitie si o insereaza imediat dupa o alta pozitie,
    translatand restul elementelor.
    """
    poz = np.random.choice(range(n), 2, replace="False")
    p1, p2 = sorted(poz)

    y = x.copy()
    # Elementul de la p2 este mutat langa p1
    val_de_mutat = x[p2]
    rest_elemente = np.delete(x, p2)
    y = np.insert(rest_elemente, p1 + 1, val_de_mutat)
    return y

def m_amestec(x: np.ndarray, n:  int) -> np.ndarray:
    """
    Mutatia prin Amestec (Scramble Mutation): Selecteaza un segment si ii amesteca
    elementele aleatoriu.
    """
    poz = np.random.choice(range(n), 2, replace=False)
    p1, p2 = sorted(poz)

    y = x.copy()
    segment = y[p1:p2 + 1]
    np.random.shuffle(segment)
    y[p1:p2+1] = segment
    return y
