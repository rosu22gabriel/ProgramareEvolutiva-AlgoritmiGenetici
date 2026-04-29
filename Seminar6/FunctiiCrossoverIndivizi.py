import numpy as np
from typing import List, Tuple, Set, Union

# =============================================================================
# 1. CROSSOVER VECTORI BINARI SAU ÎNTREGI
# =============================================================================

def crossover_unipunct(x: np.ndarray, y: np.ndarray, n:int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Realizeaza recombinarea unipunct: taie parintii intr-un punct si schimba cozile.
    """
    # i este punctul de taiere (index intre 1 si n - 1)
    i = np.random.randint(1, n - 1)
    c1, c2 = x.copy(), y.copy()

    # Copilul 1 primeste capul de la x si coada de la y
    c1[i:n] = y[i:n]
    # Copilul 2 primeste coada de la x si capul de la y
    c2[i:n] = x[i:n]
    return c1, c2

def crossover_uniform(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pentru fiecare gena, decide aleator (50/50) daca parintii schimba valorile intre ei.
    """
    c1, c2 = x.copy(), y.copy()
    for i in range(n):
        if np.random.randint(0, 2) == 1:
            c1[i], c2[i] = y[i], x[i]
    return c1, c2

# =============================================================================
# 2. CROSSOVER PERMUTĂRI (PMX, OCX, CX, ECX)
# Speciali deoarece copiii trebuie să rămână permutări (fără duplicate)
# =============================================================================

def crossover_PMX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Partially Mapped Crossover: Pastreaza ordinea si pozitia unor elemente prin mapari.
    """
    poz = np.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = np.random.randint(0, n, 2)
    p1, p2 = np.min(poz), np.max(poz)

    c1 = PMX(x, y, n, p1, p2)
    c2 = PMX(y, x, n, p1, p2)
    return c1, c2

def PMX(x: np.ndarray, y: np.ndarray, n: int, p1: int, p2: int) -> np.ndarray:
    """Implementarea logica a operatorului PMX pentru un singur copil."""
    c = -np.ones(n, dtype=int)
    # 1. Copiaza secventa de recombinare de la parintele principal
    c[p1:p2+1] = x[p1:p2+1]

    # 2. Mapare elemente din y care nu sunt deja in secventa copiata
    for i in range(p1, p2+1):
        a = y[i]
        if a not in c:
            curent = i
            plasat = False
            while not plasat:
                b = x[curent]
                # Cauta unde se afla elementul de mapare in celalalt parinte
                [poz] = [j for j in range(n) if y[j] == b]
                if c[poz] == -1: # Daca pozitia e libera, plaseaza elementul
                    c[poz] = a
                    plasat = True
                else: # Altfel, continua maparea (lant de mapari)
                    curent = poz

    # 3. Completeaza restul pozitiilor ramase cu elemente din y
    z = [y[i] for i in range(n) if y[i] not in c]
    poz_libere = [i for i in range(n) if c[i] == -1]
    for i in range(len(poz_libere)):
        c[poz_libere[i]] = z[i]
    return c

def crossover_OCX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Order Crossover: Se concentreaza pe pastrarea ordinii relative a elementelor."""
    poz = np.random.randint(0, n, 2)
    while poz[0] == poz[1]:
        poz = np.random.randint(0, n, 2)
    p1, p2 = np.min(poz), np.max(poz)

    c1 = OCX(x, y, n, p1, p2)
    c2 = OCX(y, x, n, p1, p2)
    return c1, c2

def OCX(x: np.ndarray, y: np.ndarray, n: int, p1: int, p2: int) -> np.ndarray:
    """Implementare logica a operatorului OCX."""
    # Extrage secventa centrala
    c_central = [x[i] for i in range(p1, p2+1)]
    # Listeaza elementele din y incepand de la p2, excluzand ce avem deja in c_central
    z1 = [y[i] for i in range(p2, n) if y[i] not in c_central]
    z2 = [y[i] for i in range(p2) if y[i] not in c_central]
    z = np.append(z1, z2)

    # Reasambleaza permutarea circular
    c_final = [z[i] for i in range(n - p2 -1)]
    c_inceput = [z[i] for i in range(n - p2 - 1, len(z))]

    return np.concatenate([c_inceput, c_central, c_final])


def crossover_CX(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Cycle Crossover: Identifica ciclurile intre parinti pentrua evita duplicatele"""
    ciclu = cicluri(x, y, n)
    c1, c2 = x.copy(), y.copy()
    for i in range(n):
        # Interschimba doar elementele din ciclurile cu index par
        if ciclu[i] % 2 == 0:
            c1[i], c2[i] = y[i], x[i]
    return c1, c2

def cicluri(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Identifica caror cicluri apartin genele (folosit de CX)."""
    index = 1
    ciclu = np.zeros(n)
    while np.any(ciclu == 0):
        i = np.where(ciclu == 0)[0][0]
        a = x[i]
        ciclu[i] = index
        b = y[i]
        while b != a:
            j = np.where(x == b)[0][0]
            ciclu[j] = index
            b = y[j]
        index += 1
    return ciclu

# =============================================================================
# 3. EDGE RECOMBINATION OPERATOR (ECX)
# =============================================================================

def constr_tabel(x: np.ndarray, y: np.ndarray, n: int) -> List[List[str]]:
    """Construiește o tabelă de adiacență a vecinilor pentru fiecare oraș/element."""
    muchii = [[] for _ in range(n)]
    # Bordează vectorii pentru a trata natura circulară a permutării
    x_ext = np.pad(x, (1, 1), mode='wrap')
    y_ext = np.pad(y, (1, 1), mode='wrap')

    for i in range(1, n + 1):
        oras = x_ext[i]
        j = np.where(y_ext == oras)[0][0]
        # Vecinii din ambii părinți
        vx = {x_ext[i - 1], x_ext[i + 1]}
        vy = {y_ext[j - 1], y_ext[j + 1]}

        comune = vx & vy
        doar_x = vx - vy
        doar_y = vy - vx

        # 'n+' înseamnă că muchia există în ambii părinți (prioritate mai mare)
        muchii[oras] = [f"{v}+" for v in comune] + [str(v) for v in doar_x] + [str(v) for v in doar_y]
    return muchii


def ECX(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """Edge Crossover: Încearcă să păstreze cât mai multe muchii (conexiuni) existente."""
    muchii = constr_tabel(x, y, n)
    z = np.zeros(n, dtype=int)
    ales = np.zeros(n, dtype=bool)

    a = x[0]  # Începe cu primul element din x
    for i in range(n):
        z[i] = a
        ales[a] = True
        # Elimină elementul curent din toate listele de vecini
        for lista in muchii:
            if str(a) in lista: lista.remove(str(a))
            if f"{a}+" in lista: lista.remove(f"{a}+")

        lp = muchii[a]
        if i < n - 1:
            if not lp:  # Dacă nu mai are vecini nevizitați, alege unul aleator
                a = np.random.choice([idx for idx, v in enumerate(ales) if not v])
            else:  # Alege vecinul cu cele mai puține muchii rămase
                a = alege(lp, muchii, n)
    return z


def alege(lp: List[str], muchii: List[List[str]], n: int) -> int:
    """Alege următoarea alelă bazat pe euristică (puțini vecini rămași sau muchii comune)."""
    # Verifică dacă există muchii comune ('+')
    for m in lp:
        if '+' in m: return int(m.replace('+', ''))

    # Altfel, alege vecinul care are el însuși cea mai scurtă listă de vecini
    lungimi = [len(muchii[int(m)]) for m in lp]
    idx_min = np.argmin(lungimi)
    return int(lp[idx_min])

# =============================================================================
# 4. CROSSOVER VECTORI NUMERE REALE (Aritmetic)
# =============================================================================

def crossover_singular(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Doar o singura gena este inlocuita cu media ponderata a parintilor"""
    i = np.random.randint(0, n)
    c1, c2 = x.copy(), y.copy()
    c1[i] = alpha * x[i] + (1 - alpha) * y[i]
    c2[i] = alpha * y[i] + (1 - alpha) * x[i]
    return c1, c2

def crossover_simplu(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Toate genele după un punct k sunt înlocuite cu media ponderată."""
    i = np.random.randint(0, n)
    c1, c2 = x.copy(), y.copy()
    for j in range(i, n):
        c1[j] = alpha * x[j] + (1 - alpha) * y[j]
        c2[j] = alpha * y[j] + (1 - alpha) * x[j]
    return c1, c2

def crossover_total(x: np.ndarray, y: np.ndarray, n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """Toate genele sunt rezultatul mediei ponderate între părinți."""
    c1 = alpha * x + (1 - alpha) * y
    c2 = (1 - alpha) * x + alpha * y
    return c1, c2