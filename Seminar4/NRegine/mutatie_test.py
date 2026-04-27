import numpy as np
# Importam functia specifica de mutatie din biblioteca
from FunctiiMutatieIndivizi  import m_perm_inserare
import matplotlib.pyplot as grafic
from typing import List, Tuple

# f. obiectiv
def foNR(x: np.ndarray) -> int:
    """
    Calculeaza calitatea (fitness) pentru problema reginelor.

    I: x - individul reprezentat ca permutare NumPy.
    E: c - numarul de perechi de regine care NU se ataca pe diagonala
    """
    n = x.size
    # Se foloseste o lista comprehensiva pentru a verifica conditia de atac
    # |i - j| == |x[i] - x[j]| inseamna ca reginele sunt pe aceeasi diagonala.
    c = len([(i, j) for i in range(n - 1) for j in range(i + 1, n) if abs(i - j) != abs(x[i] - x[j])])

    return c

# genereaza populatia initiala
def gen(n: int, dim: int) -> np.ndarray:
    """
    Initializeaza populatia cu permutari aleatorii.

    I: n - dimensiunea tablei, dim - marimea populatiei
    E: pop - matricea populatiei (dim x n+1).
    """
    # pop[i, n] stocheaza calitatea calculata imediat dupa generare
    pop = np.zeros((dim, n + 1), dtype=int)
    for i in range(dim):
        pop[i, :n] = np.random.permutation(n)
        pop[i, n] = foNR(pop[i, :n])

    # Reprezentare grafica initiala cu verde
    grafic.plot(pop[:, n], "go", markersize=11, label="Initial")
    return pop

# mutatie asupra populatiei de copii
def mutatie_populatie(pop: np.ndarray, dim: int, n: int, pm: float) -> np.ndarray:
    """
    Aplica operatorul de mutatie prin inserare pe intreaga populatie

    I: pop - populatia curenta, pm - probabilitatea de mutatie (ex: 0.2)
    E: mpop - populatie rezultata dupa aplicarea mutatiilor
    """
    mpop = pop.copy()
    for i in range(dim):
        # Generam un numar aleator pentru a decide daca individul sufera o mutatie
        r = np.random.uniform(0, 1)
        if r <= pm:
            # Apelam functia din biblioteca care reordoneaza genele prin inserare
            # m_perm_inserare preia genele (0:n) si returneaza o noua permutare.
            x = m_perm_inserare(mpop[i, :n], n)
            mpop[i, :n] = x.copy()
            # Este esential sa recalculam calitatea dupa ce genele au fost modificate
            mpop[i, n] = foNR(x)

    # Reprezentare grafica dupa mutatie cu rosu (pentru comparatie)
    grafic.plot(mpop[:, n], "ro", markersize=7, label="Dupa mutatie")
    return mpop

if __name__ == "__main__":
    # Exemplu: 14 indivizi pentur o tabal de 12 x 12
    p = gen(12, 14)

    # Aplicam mutatia cu o probabilitate de 20%
    grafic.title("Efectul mutatie prin inserare (FunctiiMutatieIndivizi)")
    grafic.xlabel("Index Individ")
    grafic.ylabel("Calitate (Fitness)")
    grafic.legend()
    grafic.show()
