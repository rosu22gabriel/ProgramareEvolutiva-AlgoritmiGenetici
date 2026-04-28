import numpy as np
import matplotlib.pyplot as grafic
from typing import List, Union

# --- Enunt Cerinta ---
# Fie f: Pn -> N, p apartine Pn, f(p) = |{(i, j) | i < j, p[i] < p[j]| functia obiectiv a unei probleme de maxim,
# unde Pn desemneaza multimea permutarilor de n elemente.
# a) Scrieti o functie pentru generarea aleatoare a unei populatii de dimensiune dim,
#   unde calitatea este memorata la sfarsitul fiecarei reprezentari.
# b) Pentru o probabilitate pm, scrieti o functie de mutatie prin interschimbare
#   pentru a obtine o noua populatie popm.
# ---------------------

def fitness(p: np.ndarray) -> int:
    """
    Calculeaza valoarea functiei obiectiv f(p) pentru o permutare.
    Numara perechile (i, j) unde i < j si p[i] < p[j]

    Args:
        p: Un vector numpy reprezentand o permutare.

    Returns:
        Valoarea fitness (numarul de inversiuni in sensul problemei de maxim)
    """
    n = p.size
    # Se utilizeaza list comprehension pentru a identifica perechile care respecta conditia
    val = len([(i, j) for i in range(n - 1) for j in range(i + 1, n) if p[i] < p[j]])
    return val

def cerinta_a(dim: int, n: int) -> np.ndarray:
    """
    Genereaza populatia initiala formata din permutari aleatorii.

    Args:
        dim: Numarul de indivizi din populatie.
        n: Numarul de elemente din permutare. (Pn)

    Returns:
        O matrice de dimensiune [dim x n+1] unde ultima coloana este fitness-ul.
    """
    # Alocare spatiu: n gene +  coloana pentru fitness
    populatie = np.zeros([dim, n + 1], dtype="int")

    for i in range(dim):
        # Generare permutare aleatorie pentru primii n indici
        populatie[i, :n] = np.random.permutation(n)
        # Calcul si stocare fitness pe ultima pozitie
        populatie[i, n] = fitness(populatie[i, :n])

    # Vizualizarea calitatii populatiei initiale
    grafic.plot([populatie[i][-1] for i in range(dim)], "bs", markersize=12, label="Populatie initiala")
    return populatie

def interschimbare(p: np.ndarray) -> np.ndarray:
    """
    Realizeaza mutatia unui singur individ prin interschimbarea a doua pozitii aleatorii.

    Args:
         p: Permutarea originala (fara fitness).

    Returns:
        Permutarea rezultata dupa interschimabre.
    """
    n = len(p)
    # Alegerea a doua puncte diferite pentru interschimbare
    i = np.random.randint(0, n - 1)
    j = np.random.randint(i + 1, n)

    r = p.copy()
    # Interschimbarea valorilor intre pozitiile i si j
    r[i], r[j] = p[j], p[i]

    print(f"\nInterschimbare in permutarea {p}, pm: float")

def cerinta_b(populatie: np.ndarray, pm: float) -> np.ndarray:
    """
    Aplica operatorul de mutatie asupra intregii populatii cu probabilitatea pm

    Args:
        populatie: Matricea populatiei actuale (include fitness)
        pm: Probabilitatea de mutatie (0.0 - 1.0)

    Returns:
        Noua populatie rezultata (popm).
    """
    dim = populatie.shape[0]
    populatie_m = populatie.copy()

    for i in range(dim):
        # Generam un numar aleator pentru a decide daca aplicam mutatia
        r = np.random.uniform(0, 1)
        if r <= pm:
            # Extragem genele (fara fitness), aplicam mutatia si actualizam fitness-ul
            populatie_m[i, :-1] = interschimbare(populatie[i, :-1])
            populatie_m[i, -1] = fitness(populatie_m[i, :-1])

    # Vizualizarea calitatii dupa mutatie pentru comparatie
    grafic.plot([populatie_m[i][-1] for i in range(dim)], "rs", markersize=9, label="Populatia dupa mutatie")

if __name__ == "__main__":
    # Configuratii experimentale
    n_elemente = 8
    dimensiune_populatie = 10
    probabilitate_mutatie = 0.2

    # Executie Cerinta A: Initializare
    pop_init = cerinta_a(dimensiune_populatie, n_elemente)
    print("Populatia initiala (Gene | Fitness)")
    print(pop_init)

    # Executie Cerinta B: Mutatie
    pop_dupa_mutatie = cerinta_b(pop_init, probabilitate_mutatie)

    # Afisare grafic
    grafic.title("Evolutia Fitness-ului: Initial vs Mutat")
    grafic.xlabel("Index Individ")
    grafic.ylabel("Valoare Fitness")
    grafic.legend()
    grafic.show()

