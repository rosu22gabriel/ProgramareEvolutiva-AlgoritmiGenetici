import numpy as np
import copy
from typing import List, Tuple, Union

# --- Enunt Cerinta (Problema 2 din PDF) ---
# Fie f: [-1, 1] x [0, 1] x [-2, 1] -> R,
# f(x1, x2, x3) = 1 + sin(2 * x1 - x3) + cos(x2) functia obiectiv a unei probleme de maxim.
# Reprezentarea este reala (genotipul este identic cu fenotipul).
#
# a. Scrieti o functie pentru generarea aleatoare a unei populatii 'pop' de dimensiune
#    Calitatea fiecarui individ este memorata la sfarsitul vectorului.
# b. Pentru o probabilitate pc, scrieti o functie de recombinare utilizand
#    operatorul de incrucisare aritmetica totala.
# ------------------------------------------

def fitness(x: np.ndarray) -> float:
    """
    Calculeaza valoarea functiei obiectiv pentru un individ.
    Conform PDF: f(x1, x2, x3) = 1 + sin(2*x1 - x3) + cos(x2)
    """
    return 1 + np.sin(2 * x[0] - x[2]) + np.cos(x[1])

def cerinta_a(dim: int, a: List[float], b: List[float]) -> np.ndarray:
    """
    Genereaza populatia initiala cu valori reale in intervalele specificate.

    Args:
        dim: Numarul de indivizi.
        a: Limitele inferioare pentru cele 3 variabile [-1, 0, -2].
        b: Limitele superioare pentru cele 3 variabile [1, 1, 1].
    """
    # Matrice de dimensiune dim x 4 (3 gene + o coloana fitnes)
    populatie = np.zeros([dim, 4])
    for i in range(dim):
        # Generam 3 numere reale uniform distribuite intre limitele a si b
        populatie[i, :3] = np.random.uniform(a, b, 3)
        # Evaluam individul
        populatie[i, 3] = fitness(populatie[i, :3])
    return populatie


def aritmetica_totala(x: np.ndarray, y: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Operatorul de Crossover Aritmetic Total:
    Creeaza doi copii prin combinarea liniara a intregului vector de gene.
    Formula: Copil = alpha * Parinte1 + (1 - alpha) * Parinte2.

    Args:
        alpha: Coeficient de ponderare (de obicei intre 0 si 1).
    """
    r1 = alpha * x + (1 - alpha) * y
    r2 = alpha * y + (1 - alpha) * x
    return r1, r2


def cerinta_b(populatie: np.ndarray, pc: float, alpha: float) -> np.ndarray:
    """
    Realizeaza recombinarea populatiei la nivel de perechi.
    Daca nu are loc crossover, copiii sunt identici cu parintii (recombinare asexuata).
    """
    dim = populatie.shape[0]
    # Copiem populatia oentru a genera urmasii
    populatie_c = populatie.copy()

    print("\n--- Proces de Recombinare Aritmetica ---")
    for i in range(0, dim - 1, 2):
        r = np.random.uniform(0, 1)
        if r <= pc:
            # Parintii (doar primele 3 coloane)
            p1 = populatie[i, :3]
            p2 = populatie[i+1, :3]

            print(f"Incrucisare intre {p1} si {p2}")

            # Aplicam crossover aritmetic pe toate genele
            c1, c2 = aritmetica_totala(p1, p2, alpha)

            # Salvam copiii si le calculam fitness-ul
            populatie_c[i, :3], populatie_c[i+1, :3] = c1, c2
            populatie_c[i, 3] = fitness(c1)
            populatie_c[i+1, 3] = fitness(c2)

            print(f"Rezulta copii cu fitness: {populatie_c[i, 3]:.4f} si {populatie_c[i+1, 3]:.4f}")
        else:
            print("Recombinare asexuata pentru aceasta pereche.")

    return populatie_c

if __name__ == "__main__":
    # Parametri conform cerintei si bunelor practici
    DIM = 10
    PC = 0.7
    ALPHA = 0.25 # Ponderea parintilor in copii

    # Limitele intervalelor pentru cele 3 variabile (x1, x2, x3)
    MIN_VALS = [-1.0, 0.0, -2.0]
    MAX_VALS = [1.0, 1.0, 1.0]

    # 1. Initializare
    pop_init = cerinta_a(DIM, MIN_VALS, MAX_VALS)
    print("Populatia initiala (Variabile | Fitness):")
    print(pop_init)

    # 2. Recombinare
    pop_urmasi = cerinta_b(pop_init, PC, ALPHA)

    # Afisare cel mai bun individ gasit
    cel_mai_bun = pop_urmasi[np.argmax(pop_urmasi[:, 3])]
    print(f"\nCel mai bun individ dupa crossover: {cel_mai_bun[:3]}")
