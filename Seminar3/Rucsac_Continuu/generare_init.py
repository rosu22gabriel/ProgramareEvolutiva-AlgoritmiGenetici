# AICI O POPULATIE ESTE O LISTA DE INDIVIZI, UN INDIVID ESTE O LISTA
# DE ELEMENTE IN [0,1] COMPLETATA CU VALOAREA

import matplotlib.pyplot as grafic
import numpy as np
import random
from typing import List, Tuple, Union

# Un individ este o lista de float-uri (genele) plus calitatea la final
Individ = List[float]
Populatie = List[Individ]

# verifica fezabilitatea alegerii x si calculeaza si f. obiectiv
def ok(x: List[float], c: np.ndarray, v: np.ndarray, maxim: float) -> Tuple[bool, float]:
    """
    Evalueaza o solutie in spatiul continuu [0, 1].

    Args:
        x: Vector de fractiuni ale obiectelor (ex: 0.5 inseamna jumatate de obiect).
        c: Vectorul costurilor obiectelor.
        v: Vectorul valorilor obiectelor.
        maxim: Capacitatea rucsacului.
    """
    # Calculam valoarea si costul folosind list comprehension (specific pentru liste)
    val = sum(x[i] * v[i] for i in range(len(x)))
    cost = sum([x[i] * c[i] for i in range(len(x))])

    # Returnam daca este valida si valoare fitness-ului
    return cost <= maxim, val

# genereaza populatia initiala
def gen(fc: str, fv: str, maxim: float, dim: int) -> Populatie:
    """
    Creeaza populatia initiala sub forma de lista de liste.
    Args:
        fc, fv: Caile catre fisierele de date.
        maxim: Capacitatea maxima admisa.
        dim: Dimensiunea populatiei (numarul de indivizi).
    """
    # Citim datele folosind numpy pentru eficienta
    c = np.genfromtxt(fc)
    v = np.genfromtxt(fv)
    n = len(c)

    # Initializam o lista goala pentru populatie
    pop: Populatie = []

    for i in range(dim):
        gata = False
        while gata == False:
            # Generare Continua:
            # random.uniform(0, 1) genereaza un numar real intre 0 si 1
            x = [random.uniform(0, 1) for _ in range(n)]
            gata, val = ok(x, c, v, maxim)

        # Concatenare Liste:
        # x + [val] creeaza o noua lista care contine genele si fitness-ul la final
        x_cu_fitness = x + [val]

        # Adaugam individul complet ca element in lista populatiei
        pop.append(x_cu_fitness)

    # Vizualizarea diversitatii initiale
    reprezinta_pop(pop, dim, n)
    return pop

# desenarea populatiei prin punctele (indice individ, calitate)
def reprezinta_pop(pop: Populatie, dim: int, n: int) -> None:
    """
    Reprezentare grafica a fitness-ului fiecarui individ din populatie.
    """
    x_axis = [i for i in range(dim)]
    # Extragem ultimul element (index n) din fiecare sub-lista pentru axa Y
    y_axis = [pop[i][n] for i in range(dim)]

    grafic.plot(x_axis, y_axis, "gs", markersize=11)
    grafic.title("Calitatile indivizilor generati (Varianta Continua)")
    grafic.xlabel("Index individ")
    grafic.ylabel("Calitate (Valoare)")
    grafic.show()


if __name__ == "__main__":
    # Parametri: date din fisiere, capacitate 50, populatie de 10 indivizi
    p = gen("cost.txt", "valoare.txt", 50, 10)

    print("\n--- Analiza Populatiei Initiale ---")
    for element in p:
        # Formatam genele pentru a fi afisate cu 3 zecimale (pentru lizibilitate)
        # element[:-1] inseamna "toate elementele in afara de ultimul"
        list_format = [float(f"{el: .3f}") for el in element[:-1]]
        # element[-1] este fitness-ul
        print(f"Individul (fractiuni): {list_format} | Calitate: {element[-1]:.4f}")