import matplotlib.pyplot as grafic
import numpy as np
from typing import Tuple

# f. obiectiv
def foTSP(p: np.ndarray, c: np.ndarray) -> float:
    """
    Calculeaza fitness-ul pentru Problema Comis-Voiajorului (TSP),
    Fitness-ul este definit ca 100 / cost_total pentru a transforma o
    problema de minimizare (cost) intr-una de maximizare (calitate).

    Args:
        p: Permutare reprezentand ordinea vizitarii oraselor.
        c: Matricea de adiacenta (costuri/distante intre orase).
    """
    n = p.size
    val = 0
    # Parcurgem traseul si adunam costul intre orase consecutive
    for i in range(n - 1):
        val += c[p[i], p[i + 1]]

    # Adaugam drumul de intoarcere de la ultimul oras la primul (circuit inchis)

    # Intr-un AG, de obicei vrem sa MAXIMIZAM. De aceea folosim 100 / val.
    # Cu cat distanta e mai mica, cu atat calitatea e mai mare.
    return 100 / val

# figurarea populatiei prin punctele (indice individ, calitate)
def reprezinta_pop(val: np.ndarray, dim: int) -> None:
    """
        Afiseaza distributia calitatii in populatia initiala.
    """
    x = [i for i in range(dim)]
    y = [val[i] for i in range(dim)]

    # "ro" - puncte de culoare rosie sub forma de cerc
    grafic.plot(x, y, "ro", markersize=7)
    grafic.title("Calitatile indivizilor generati in populatia initiala.")
    grafic.xlabel("Index individ")
    grafic.ylabel("Calitate (100 / Distanta)")
    grafic.show()

# genereaza populatia initiala
def gen(fc: str, dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genereaza populatia ca matrice de permutari si calitatile ca vector separat.

    Args:
        fc: Fisierul cu matricea de costuri (n x n).
        dim: Dimensiunea populatiei.
    """

    # Citim matricea de costuri
    c = np.genfromtxt(fc)
    # n este numarul de orase (dimensiunea problemei)
    n = c.shape[0]

    # Matrice pentru indivizi (fiecare rand este o ruta)
    pop = np.zeros((dim, n), dtype=int)
    # Vector separat pentru fitness
    val = np.zeros(dim, dtype = float)

    for i in range(dim):
        # Fiecare individ este o permutare aleatorie a oraselor 0...n-1
        pop[i] = np.random.permutation(n)
        # Evaluam ruta generata
        val[i] = foTSP(pop[i, :n], c)

    reprezinta_pop(val, dim)
    return pop, val

if __name__ == "__main__":
    # Test pentru o populatie de 30 de indivizi
    # Nota: Fisierul costuri.txt trebuie sa contina o matrice patratica n x n
    try:
        p, v = gen("costuri.txt", 30)
        print(f"Dimensiunea problemei (orase): {p.shape[1]}")
        for i in range(p.shape[0]):
            # Afisam ruta si calitatea ei (100 / distanta)
            print(f"Individul {i}: {p[i]} | Calitate: {v[i]:.4f}")
    except OSError:
        print("Eroare: Fisierul 'costuri.txt' nu a fost gasit.")