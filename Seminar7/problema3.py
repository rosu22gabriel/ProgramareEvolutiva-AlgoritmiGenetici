import random
import copy
from typing import List, Tuple, Union

# --- Enunt Cerinta 3 (Problema 3 din PDF) ---
# Fie functia f(x) = suma(xi) pentru i = 1...7, unde x = (x1, ..., x7) este un vector binar.
# Aceasta este o problema de maxim (cunoscuta sub numele de OneMax).
#
# a. Scrieti o functie pentru generarea aleatoare a unei populatii "pop" de dimensiune 'dim'.
#    Calitatea fiecarui individ este memorata la sfarsitul fiecarei reprezentari.
# b. Pentru o probabilitate pc, scrieti o functie de recombinare utilizand
#    operatorul de incrucisare multi-punct cu 2 puncte de incrucisare.
# --------------------------------------------

def fitness(x: List[int]) -> int:
    """
    Calculeaza calitatea individului (numarul de biti de 1).
    Conform PDF: f(x) = suma(xi)
    """
    return sum(x)


def cerinta_a(dim: int) -> List[List[Union[int, float]]]:
    """
    Genereaza populatia initiala formata din vectori binari de lungime 7.
    """
    populatie = []
    for i in range(dim):
        # Generam 7 biti aleatorii (0 sau 1)
        individ = random.choices([0, 1], k = 7)
        # Calculam fitness-ul si il adaugam la final
        individ.append(fitness(individ))
    return populatie


def recombinare_2(x: List[int], y: List[int]) -> Tuple[List[int], List[int]]:
    """
    Operator de incrucisare cu 2 puncte de taiere (2-point crossover).
    Selecteaza doua puncte aleatorii si interschimba segmentul dintre ele.
    """
    print(f"\nRecombinare intre parintii {x} si {y}")
    copil1, copil2 = x.copy(), y.copy()

    # Alegem doua puncte de taiere distincte in intervalul [1, 6]
    # Folosim sample pentru a ne asigura ca sunt unice si sorted pentru ordine
    i, j = sorted(random.sample(range(1, 7), 2))
    print(f"Punctele de taiere selectate: {i}, {j}")

    # Interschimbam segmentul cuprins intre indicii i si j
    copil1[i:j] = y[i:j]
    copil2[i:j] = x[i:j]

    print(f"Rezulta copiii: {copil1} si {copil2}")
    return copil1, copil2


def cerinta_b(parinti: List[List[Union[int, float]]], pc: float) -> List[List[Union[int, float]]]:
    """
    Realizeaza recombinarea la nivelul intregii populatii.
    Indivizii sunt parcursi in perechi, iar crossover-ul are loc cu probabilitatea pc.
    """
    copii = copy.deepcopy(parinti)
    dim = len(parinti)

    for i in range(0, dim - 1, 2):
        raspuns = random.uniform(0, 1)
        if raspuns < pc:
            # Extragem genotipurile (fara ultima valoare care este fitness-ul)
            parinte1 = parinti[i][:-1]
            parinte2 = parinti[i+1][:-1]

            # Aplicam operatorul de recombinare
            c1, c2 = recombinare_2(parinte1, parinte2)

            # Calculam noile calitati pentru urmasi
            c1.append(fitness(c1))
            c2.append(fitness(c2))

            # Actualizam populatia de copii
            copii[i] = c1
            copii[i+1] = c2
        else:
            print(f"\nPerechea {i}-{i+1}: Recombinare asexuata (copiere parinti.)")

    return copii

if __name__ == "__main__":
    # Pareametri algoritmului
    DIM_POPULATIE = 10
    PROB_RECOMBINARE = 0.8

    # 1. Generare populatie initiala
    pop_init = cerinta_a(DIM_POPULATIE)

    # 2. Aplicare recombinare
    pop_copii = cerinta_b(pop_init, PROB_RECOMBINARE)

    # Afisare rezultate
    print("\n" + "="*30)
    print("POPULATIA DE PARINTI")
    for ind in pop_init:
        print(ind)

    print("\nPOPULATIA DE COPII")
    for ind in pop_copii:
        print(ind)