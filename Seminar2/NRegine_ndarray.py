import numpy as np
import matplotlib.pyplot as grafic
from typing import Tuple, List

def f_ob(p: np.ndarray) -> float:
    """
    Functia obiectiv: Calculeaza calitatea unei asezari.
    Fiecare pereche de regine care NU se ataca pe diagonala adauga 1 la calitate.
    """
    n = p.size
    # Calitatea maxima teoretica: n combinat de 2 (perechi posibile)
    calitate = n * (n - 1) / 2
    for i in range(n - 1):
        for j in range(i + 1, n):
            # Verifica atacul pe diagonala
            if abs(i - j) == abs(p[i] - p[j]):
                calitate -= 1
    return calitate


def vecini(p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genereaza toti vecinii posibil folosind transpozitii (interschimbari)
    """
    n = p.size
    # Pre-alocam spatiu in memorie pentru matricea de vecini si vectorul de calitati
    # Este mult mai rapid decat np.append
    nr_vecini = round(n * (n - 1) / 2)
    Vecini_p = np.zeros([nr_vecini, n], dtype='int')
    Calitati_vecini_p = np.zeros(nr_vecini, dtype='int')

    # Contor pentru a urmari randul curent in matrice de vecini
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            v = p.copy()
            # Transpozitie: schimbam doua elemente intre ele
            v[i], v[j] = v[j], v[i]
            Vecini_p[k] = v
            Calitati_vecini_p[k] = f_ob(v)
            k += 1

    return Vecini_p, Calitati_vecini_p

def HC(n: int) -> Tuple[np.ndarray, float]:
    """
    Hill Climbing pentru un punct initial generat aleator ca permutare.
    """
    # np.random.permutation genereaza un aranjament aleator al numerelor 0..n-1
    s = np.random.permutation(n)
    v_s = f_ob(s)
    local = False

    while not local:
        Vecini, Calitati = vecini(s)
        # np.argmax gaseste indexul valorii maxime dintr-un vector
        index = np.argmax(Calitati)

        if v_s < Calitati[index]:
            # Daca cel mai bun vecin e mai bun decat starea curenta, ne mutam
            s = Vecini[index].copy()
            v_s = Calitati[index]
        else:
            # Altfel, am atins un maxim local (varf)
            local = True
    return s, v_s

def apel_HC(n: int, k: int) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """
    Ruleaza algoritmul HC de k ori (Random Restart) pentru a evita blocarea in optime locale slabe.
    """
    # Stocam rezultatele fiecare restartari intr-o matrice si un vector
    Puncte = np.zeros([k, n], dtype="int")
    Valori = np.zeros(k, dtype="int")

    for i in range(k):
        x, v_t = HC(n)
        Puncte[i] = x.copy()
        Valori[i] = v_t

    # Extragem cea mai buna solutie dintre toate cele k incercari
    v_max = Valori.max()
    p_max = Puncte[np.argmax(Valori)]

    print("-" * 30)
    print("Maximul calculat:", v_max)
    print("Alegerea corespunzatoare", p_max)
    print("-" * 30)

    arata(p_max, v_max)
    return p_max, v_max, Puncte, Valori


def arata(sol: np.ndarray, val: float) -> None:
    """
    Functie pentru vizualizarea tablei de sah si a pozitiilor reginelor.
    """
    n = sol.size
    minim = n * (n - 1) / 2 - val # Cate conflicte (atacuri) au ramas

    t1 = "Analiza N-Regine\n"
    t2 = "Starea finala gasita "
    t4 = f" (Conflicte {int(minim)} vs 0"

    t3 = "nu este optima" if minim > 0 else "este OPTIMA"
    titlu = t1 + t2 + t3 + t4
    fig = grafic.figure(figsize=(6, 6))
    ax = fig.gca()

    # Transformam indicii in coordonate de afisare (centrul patratelelor)
    x_coords = [i + 0.5 for i in range(n - 1, -1, -1)]
    y_coords = [sol[i] + 0.5 for i in range(n)]

    grafic.plot(y_coords, x_coords, 'r*', markersize=15)

    # Configurare grid pentru a simula tabla de sah
    grafic.xticks(range(n + 1))
    grafic.yticks(range(n + 1))
    grafic.grid(True, which='both', color='k', linestyle='-', linewidth=1)
    ax.set_aspect('equal')
    grafic.title(titlu)
    grafic.show()

if __name__ == '__main__':
    # Test pentru 8 regine cu 200 de restartari aleatoare
    sol, val, P, C = apel_HC(8, 200)
