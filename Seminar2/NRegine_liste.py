import random
import matplotlib.pyplot as grafic
from typing import List, Tuple

def f_ob(p: List[int]) -> float:
    """
    Functia obiectiv (Fitness): Calculeaza cate perechi de regine NU se ataca pe diagonala
    O solutie este optima daca fitness-ul este n*(n-1)/2 (zero atacuri).

    Args:
        p: O permutare unde p[i] este randul pe care se afla regina de pe coloana i.
    """
    n = len(p)
    # Numarul maxim de perechi posibile de regine
    nr_poz_corecte = n * (n - 1) / 2

    for i in range(n - 1):
        for j in range(i + 1, n):
            # Deoarece p este o permutare, reginele nu pot fi pe acelasi rand sau coloana.
            # Verificam doar daca sunt pe aceeasi diagonala:
            # Diferenta dintre coloane == Diferenta dintre randuri
            if abs(i - j) == abs(p[i] - p[j]):
                nr_poz_corecte -= 1

    return nr_poz_corecte


def vecini(p: List[int]) -> Tuple[List[List[int]], List[float]]:
    """
    Genereaza vecinii prin interschimbarea a doua elemente din permutare (Swap).
    Aceasta metoda pastreaza proprietatea de permutare a solutiei.
    """

    n = len(p)
    Vecini_p = []
    Calitati_vecini_p = []

    for i in range(n - 1):
        for j in range(i + 1, n):
            x = p.copy()
            # Swap: schimbam pozitia reginei de pe coloana i cu cea de pe coloana j
            x[i], x[j] = x[j], x[i]
            Vecini_p.append(x)
            Calitati_vecini_p.append(f_ob(x))

    return Vecini_p, Calitati_vecini_p

def HC(n: int) -> Tuple[List[int], float]:
    """
    Algoritmul Hill Climbing pentru un singur punct de pornire.
    """
    # Genereaza o permutare aleatorie a numerelor de la 0 la n - 1

    # Functia random.sample din biblioteca Standard Python este folosita pentrua a extrage
    # un grup de elemente unice dintr-o lista sau o secventa, fara a modifica lista originala
    # Este instrumentul ideal atunci cand ai nevoie de esantionare fara inlocuire ( o data ce un element
    # a fost ales, el nu mai poate aparea a doua oara in rezultat).

    # Sintaxa: random.sample(populatie, k)
    # populatie -> Colectia din care vrei sa extragi (lista, set, string, range, etc.)
    # k: Numarul de elemente pe care vrei sa le extragi
    s = random.sample(range(n), n)

    v_s = f_ob(s)
    local = False

    while not local:
        Vecini, Calitati = vecini(s)
        # Gaseste indexul vecinului cu calitatea maxima
        index = max(range(len(Calitati)), key=lambda i: Calitati[i])

        if v_s < Calitati[index]:
            s = Vecini[index].copy()
            v_s = Calitati[index]
        else:
            # Daca niciun vecin nu e mai bun, am ajuns la un optim local
            local = True

    return s, v_s


def apel_HC(n: int, k: int) -> Tuple[List[int], float, List[List[int]], List[float]]:
    """
    Ruleaza Hill Climbing de k ori (Random Restart) pentru a gasi solutia optima
    """
    Puncte = []
    Valori = []

    for i in range(k):
        x, v_t = HC(n)
        Puncte.append(x)
        Valori.append(v_t)

    v_max = max(Valori)
    index = Valori.index(v_max)
    p_max = Puncte[index]

    print(f"Maximul calculat: {v_max} pentru configuratia {p_max}")
    arata(p_max, v_max)

    return p_max, v_max, Puncte, Valori


def arata(sol: List[int], val: float) -> None:
    """
    Vizualizeaza asezarea reginelor pe o tabla de sah folosind coordonatele din solutie.
    """
    n = len(sol)
    # Calculam cate atacuri (erori) au ramas:
    minim = n * (n - 1) / 2 - val

    t1 = "Rezultat Hill Climbing - N-Regine\n"
    if minim > 0:
        t3 = f"NU este optim (Erori: {int(minim)}"
    else:
        t3 = "Este OPTIM (0 atacturi intre regine)"

    titlu = t1 + t3

    fig = grafic.figure(figsize=(6, 6))
    ax = fig.gca()

    # Coordonatele pentru desenare:
    # x reprezinta randul (inversat pentru afisare), y reprezinta coloana

    x_pos = [i + 0.5 for i in range(n - 1, - 1, -1)]
    y_pos = [sol[i] + 0.5 for i in range(n)]

    # Desenam reginele ca niste stele rosii
    grafic.plot(y_pos, x_pos, 'r*', markersize=15)

    # Setari pentru a face graficul sa arate ca o tabla de sah
    grafic.xticks(range(n + 1))
    grafic.yticks(range(n + 1))
    grafic.grid(True, which='both', color='k', linestyle='-', linewidth=1)
    ax.set_aspect('equal')
    grafic.title(titlu)
    grafic.show()


if __name__ == "__main__":
    # n = 8 regine, k = 200 incercari (restarts)
    sol, val, P, C = apel_HC(8, 200)
