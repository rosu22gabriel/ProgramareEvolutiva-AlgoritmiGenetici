# Aici populatia este o matrice in care fiecare linie este o permutare completata de calitatea acesteia

import numpy as np
import matplotlib.pyplot as grafic

# functie obiectiv
def foNR(x: np.ndarray) -> float:
    """
    Functie obiectiv pentru problema celor N-Regine

    Args:
        x: Un individ reprezentat ca ndarray (permutare).

    Returns:
        c: Calitatea (numarul de perechi de regine care NU se ataca)
    """
    n = x.size
    # Numarul maxim de perechi de regine care pot fi formate: n(n-1)/2
    c = n * (n - 1) / 2

    for i in range(n - 1):
        for j in range(i + 1, n):
            # Verificam daca reginele de pe coloanele i si j sunt pe aceeasi diagonala
            # (Diferenta randurilor este egala cu diferenta coloanelor)
            if abs(i - j) == abs(x[i] - x[j]):
                c -= 1

    return c

# genereaza populatia initiala
def gen(n: int, dim: int) -> np.ndarray:
    """
    Genereaza populatia initiala pentru algoritm.

    Args:
        n: Dimensiunea problemei (numarul de regine/dimensiunea tablei).
        dim: Numarul de indivizi din populatie (marimea populatiei).

    Returns:
        pop: Matrice de dimensiune dim x (n + 1).
             Ultimele coloane (index n) stocheaza fitness-ul fiecarui individ.
    """
    # Definim o matrice de zerouri: dim randuri si n + 1 coloane
    # Coloanele 0 -> n - 1 sunt genele (pozitiile reginelor)
    # Coloana n este rezerva pentru valoare functiei obiectiv
    pop = np.zeros((dim, n + 1), dtype=int)

    for i in range(dim):
        # Generam o permutare aleatorie pentru primele n coloane
        pop[i, :n] = np.random.permutation(n)

        # Calculam calitatea si o stocam pe ultima coloana (indexul n)
        # Transmitand pop[i, :n] extragem doar genele, fara coloane de calitate
        pop[i, n] = foNR(pop[i, :n])

    # Vizualizam distributia calitatii in populatia proaspat generata
    reprezinta_pop(pop, dim, n)
    return pop

# desenarea populatiei prin puncte (indice individ, calitate)
def reprezinta_pop(pop: np.ndarray, dim: int, n: int) -> None:
    """
    Creeaza un grafic pentru a vizualiza variabilitatea calitatii in populatie.
    Ajuta la intelegerea conceptului de 'Diversity' (diversitate genetica).
    """
    # Coordonata X: Indexul fiecarui individ din matrice (0, 1, 2...)
    x_axis = [i for i in range(dim)]

    # Coordonata Y: Valoare fitness de pe ultima coloana a matricei
    y_axis = [pop[i][n] for i in range(dim)]

    # Desenam punctele ca patrate verzi ("gs")
    grafic.plot(x_axis, y_axis, "gs", markersize=11)

    grafic.title("Calitatile indivizilor generati in populatia initiala")
    grafic.xlabel("Index individ (ID in matrice)")
    grafic.ylabel("Calitate (Fitness)")
    grafic.grid(True, linestyle="--", alpha=0.6)
    grafic.show()

if __name__ == "__main__":
    # Exemplu: Tabla de 8x8 cu o populatie de 30 de indivizi
    populatie_initiala = gen(8, 30)

    print("Matricea Populatiei (Linii = Indivizi, Ultima coloana = Calitate):")
    print(populatie_initiala)
