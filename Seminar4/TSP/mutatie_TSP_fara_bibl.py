import numpy as np
import matplotlib.pyplot as grafic
from typing import Tuple

# f. obiectiv
def foTSP(p: np.ndarray, c: np.ndarray) -> float:
    """
    Calculeaza fitness-ul (calitatea) unei rute TSP

    Args:
         p: Permutare a oraselor (individul).
         c: Matricea costurilor intre orase.

    Returns:
         Calitatea calculata ca 100 / cost_total.
    """

    n = p.size
    # Calculam costul total: drumul de intoarcere (ultimul -> primul) + drumul intre orase consecutive
    cost = c[p[0]][p[-1]] + sum([c[p[i]][p[i+1]] for i in range(n - 1)])
    return 100 / cost

def gen(c: np.ndarray, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genereaza populatia initiala de rute (permutari).

    Args:
        c: Matricea costurilor.
        dim: Numarul de indivizi (rute) din populatie.
    """
    n = c.shape[0]
    populatie = np.zeros([dim, n], dtype="int")
    valori = np.zeros(dim)

    for i in range(dim):
        # Fiecare individ este o permutare aleatorie a oraselor 0...n-1
        populatie[i] = np.random.permutation(n)
        valori[i] = foTSP(populatie[i], c)

    # Reprezentam fitness-ul initial cu patrate verzi
    grafic.plot(valori, "gs", markersize=11, label="Initial")
    return populatie, valori

def mutatie_inversiune(p: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Realizeaza mutatia prin inversiune (2-opt move style)
    Inverseaza ordinea elementelor intre doua puncte i si j.
    """
    n = p.size
    print(f"Individul initial {p}")

    # Selectam doua pozitii aleatorii
    i = np.random.randint(0, n - 1)
    j = np.random.randint(i + 1, n)

    print(f"Pozitiile de mutatie: {i}, {j}")

    r = p.copy()
    # Inversam secventa intre indexul i si j (inclusiv j)
    # p[k] parcurge elementele in sens invers de la j la i
    r[i:j+1] = [p[k] for k in range(j, i - 1, -1)]

    print(f"Individul dupa mutatie {r}\n")
    return r

def mutatie_populatie(pcopii: np.ndarray, vcopii: np.ndarray, c: np.ndarray, pm: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica mutatia prin inversiune asupra intregii populatii cu o probabilitate pm.
    """
    # Copiem datele pentru a nu altera populatia originala
    mpcopii = pcopii.copy()
    mvcopii = vcopii.copy()
    dim = mpcopii.shape[0]

    for i in range(dim):
        # Generam un numar aleator pentru a decide mutatia
        r = np.random.uniform(0, 1)
        if r <= pm:
            x = mpcopii[i]
            y = mutatie_inversiune(x, c)
            mpcopii[i] = y.copy()
            # Recalculam calitatea pentru noul traseu inversat
            mvcopii[i] = foTSP(y, c)

    # Reprezentam grafic fitness-ul dupa mutatie cu patrate rosii
    grafic.plot(mvcopii, "rs", markersize=7, label="Dupa mutatie")
    return mpcopii, mvcopii

if __name__ == "__main__":
    # Incarcam matricea de costuri (distante intre orase)
    try:
        costuri = np.genfromtxt("costuri.txt")
        # Generam populatie si aplicam mutatia
        p, v = gen(costuri, 14)
        o, vo = mutatie_populatie(p, v, costuri, 0.2)
        grafic.title("TSP: Mutatia prin Inversiune")
        grafic.xlabel("Index Invalid")
        grafic.ylabel("Calitate (100/Cost)")
        grafic.legend()
        grafic.show()

    except OSError:
        print("Eroare: Fisierul 'costuri.txt' nu exista.")
