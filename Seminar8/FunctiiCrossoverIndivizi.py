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
    Returneaza un vector unde fiecare pozitie indica indexul ciclului din care face parte.
    Acest operator este special deoarece asigura faptul ca fiecare gena a copilului provine
    de la unul dintre parinti din exact aceeasi pozitie, eliminand riscul de a introduce elemente
    noi sau de a strica ordinea absoluta.
    """
    index = 1
    ciclu = np.zeros(n, dtype=int)
    gata = 0
    while not gata:
        p = np.where(ciclu == 0)[0]
        if len(p) > 0:
            i = p[0]
            a = x[i]
            ciclu[i] = index
            b = y[i]
            while b != a:
                # Cautam unde se afla valoarea b in parintele x
                j = np.where(x == b)[0][0]
                ciclu[j] = index
                b = y[j]
            index += 1
        else:
            gata = 1
    return ciclu

def crossover_CX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cycle Crossover (CX): Interschimba elementele in functie de ciclurile paritatii.
    """
    ciclu = cicluri(x, y, n)
    c1, c2 = x.copy(), y.copy()
    for i in range(n):
        # Daca ciclul are index par, interschimbam genele
        if ciclu[i] % 2 == 0:
            c1[i], c2[i] = y[i], x[i]
    return c1, c2


def constr_tabel(x: np.ndarray, y: np.ndarray, n: int) -> List[List[str]]:
    """
    Construieste tabela de adiacenta (vecini) pentru ECX.
    Este unul dintre cei mai complecsi operatori pentru permutari. Se concentreaza
    pe pastrarea muchiilor (legaturilor dintre orase) mai degraba decat pe pozitia
    genelor.
    """
    muchii = [[] for _ in range(n)]
    # Ordonam (?) vectorii pentru a gestiona circularitate (ultimul e vecin cu primul)
    x_ext = np.insert(x, [0, n], [x[n - 1], x[0]])
    y_ext = np.insert(y, [0, n], [y[n - 1], y[0]])

    for i in range(1, n + 1):
        alela = x_ext[i]
        idx_y = np.where(y == alela)[0][0] + 1

        # Vecinii din ambii parinti
        vx = {x_ext[i-1], x_ext[i + 1]}
        vy = {y_ext[idx_y-1], y_ext[idx_y + 1]}

        comune = vx & vy
        doar_x = vx - vy
        doar_y = vy - vx

        # Marcam muchiile comune cu '+' conform algoritmului
        muchii[alela] = [f"{m}+" for m in comune] + [str(m) for m in doar_x] + [str(m) for m in doar_y]
    return muchii

def sterge_din_tabel(muchii: List[List[str]], alela: int):
    """Elimina o alela deja aleasa din toate listele de vecini."""
    tinta = str(alela)
    tinta_plus = f"{alela}+"
    for lista in muchii:
        if tinta in lista: lista.remove(tinta)
        if tinta_plus in lista: lista.remove(tinta_plus)

def ECX(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Edge Crossover (ECX): Construieste un copil alegand sistematic orasul cu cei mai putini vecini
    ramasi in tabela.
    """
    muchii = constr_tabel(x, y, n)
    z = np.zeros(n, dtype=int)
    ales = np.zeros(n, dtype=bool)
    curr = x[0] # Incepem cu prima gena a primului parinte
    for i in range(n):
        z[i] = curr
        ales[curr] = True
        sterge_din_tabel(muchii, curr)

        # Alegem urmatorul canditat
        lista_vecini = muchii[curr]
        if len(lista_vecini) > 0:
            # Prioritate: muchii comune (+), apoi lungimea minima a listei de vecini
            comune = [int(m[:-1]) for m in lista_vecini if "+" in m]
            if comune:
                curr = comune[0]
            else:
                # Alegem vecinul care are, la randul lui, cei mai putini vecini
                vecini_int = [int(m) for m in lista_vecini]
                lungimi = [len(muchii[v]) for v in vecini_int]
                curr = vecini_int[np.argmin(lungimi)]
        else:
            # Daca nu mai sunt vecini, alegem ceva aleator neales
            nealese = np.where(ales == False)[0]
            if len(nealese) > 0:
                curr = np.random.choice(nealese)
    return z

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

def crossover_simplu(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crossover Simplu (Real): Pastreaza genele intacte pana la punctul 'i',
    apoi aplica recombinarea aritmetica pentru restul vectorului.
    Se foloseste pentru numere reale si aplcia media ponderata doar de la un anumit punct in colo.
    """
    i = np.random.randint(0, n)
    c1, c2 = x.copy(), y.copy()
    for j in range(i, n):
        c1[j] = alpha * x[j] + (1 - alpha) * y[j]
        c2[j] = alpha * y[j] + (1 - alpha) * x[j]
    return c1, c2