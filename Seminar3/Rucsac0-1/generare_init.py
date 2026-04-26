# AICI O POPULATIE ESTE O LISTA DE INDIVIZI, UN INDIVID ESTE O LISTA
# CU 0,1 COMPLETATA CU VALOAREA

import numpy as np
import random
import matplotlib.pyplot as grafic
from typing import List, Tuple, Union

# Definirea tipurilor pentru lizibilitate
# Un individ este o listă de intregi (genele 0/1) urmată de un float (fitness-ul)
Individ = List[Union[int, float]]
Populatie = List[Individ]

# verifica fezabilitatea alegerii x si calculeaza si f. obiectiv
def ok(x: List[int], c: np.ndarray, v: np.ndarray, maxim: float) -> Tuple[bool, float]:
    """
    Verifica daca solutia binara x respecta capacitatea rucsacului.

    Args:
        x: Lista de gene (0 sau 1).
        c: Vectorul NumPy al costurilor.
        v: Vectorul NumPy al valorilor.
        maxim: Capacitatea rucsacului.
    """
    # Folosim np.dot chiar daca x este lista; Numpy va face conversia intern pentru calcul
    val = np.dot(x, v)
    cost = np.dot(x, c)
    return cost <= maxim, val

# figurarea populatiei prin punctele (indice individ, calitate)
def reprezinta_pop(pop: Populatie, dim: int, n: int) -> None:
    """
    Vizualizeaza fitness-ul fiecarui individ din populatie.
    """
    x_axis = [i for i in range(dim)]
    # Preluam calitatea de pe ultima pozitie (index n) a fiecarui individ
    y_axis = [pop[i][n] for i in range(dim)]

    grafic.plot(x_axis, y_axis, "gs", markersize=11)
    grafic.title("Calitatile indivizilor generati in populatia initiala")
    grafic.xlabel("Index individ")
    grafic.ylabel("Calitate (Fitness)")
    grafic.show()

# genereaza populatia initiala
def gen(fc: str, fv: str, maxim: float, dim: int) -> Populatie:
    """
    Genereaza populatia initiala de indivizi binari sub forma de lista de liste.

    Args:
        fc, fv: Fisierele cu datele problemei
        maxim: Limita de greutate.
        dim: Cati indivizi sa aiba populatia.
    """
    # Citim datele ca array-uri NumPy pentru calcul rapid
    c = np.genfromtxt(fc)
    v = np.genfromtxt(fv)
    n = v.size

    # Initializam populatia ca o lista goala
    pop: Populatie = []

    for i in range(dim):
        gata = False
        while gata == False:
            # random.choices(populatie, k=n) extrage k elemente din lista cu inlocuire
            # Este metoda ideala pentru a genera un sir aleator de 0 si 1
            x = random.choice([0, 1], k = n)
            gata, val = ok(x, c, v, maxim)

        # Odata gasit un individ valid, ii atasam fitness-ul
        # Spre deosebire de concatenarea listelor (x + [val]), append() modifica lista existenta
        x.append(val)

        # Adaugam individul (lista) in populatie (lista mare)
        x.append(val)

        # Adaugam individul (lista) in populatie (lista mare)
        pop.append(x)

    reprezinta_pop(pop, dim, n)
    return pop


if __name__ == "__main__":
    # Apelam functia pentru a genera o populatie de 10 indivizi
    # pentru un rucsac de capacitate 50
    p = gen("cost.txt", "valoare.txt", 50, 10)

    print("\n--- Populatia Generata ---")
    for index, element in enumerate(p):
        # element[:-1] sunt genele (0 sau 1)
        # element[-1] este fitness-ul (valoare totala)
        print(f"Individ {index}: {element[:-1]} | Calitate: {element[-1]}")
