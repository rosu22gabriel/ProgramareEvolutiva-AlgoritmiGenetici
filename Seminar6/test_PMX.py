import numpy as np
from typing import Tuple

# --- Enunt Conceptual ---
# PMX (Partially Mapped Crossover) este conceput pentru a transmite informatia de ordine
# si pozitie de la parinti la copii, fiind ideal pentru problema comis-voiajorului (TSP).
# Mecanismul de mapare garanteaza ca fiecare oras apare exact o data in noul individ.
# ------------------------

def c_PMX(x: np.ndarray, y: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """
    Functie suport care genereaza un singur copil folosind logica PMX.

    Args:
        x: Parintele principal (de la care se mosteneste secventa centrala).
        y: Parintele secundar (folosit pentru mapare si completare).
        p1, p2: Indicii care definesc limitele secventei de crossover.
    """

    n = x.size
    # Initializam copilul cu -1 pentru a marca pozitiile libere
    copil = -np.ones(n, dtype="int")

    # 1. Copiem secventa de crossover din primul parinte (mostenire directa)
    copil[p1:p2+1] = x[p1:p2 + 1]

    # 2. Maparea elementelor din parintele 2 care se afla in secventa [p1, p2]
    # dar nu au fost deja copiate in copil (conflicte de duplicat)

    # Identificam indicii din y din zona centrala a caror valori nu sunt inca in copil
    poz_alele_neplasate = [k for k in range(p1, p2+1) if not y[k] in copil]


    for p in poz_alele_neplasate:
        a = y[p] # Valoarea din parintele 2 care trebuie plasata


        # Cautam o pozitie libera pentru 'a' urmand lantul de mapari
        # p devine indexul unde valoarea x[p] se regaseste in y
        while copil[p] > -1:
            # Cautam unde se afla valoarea care ocupa deja locul in copil (din parintele y)
            valoare_conflict = copil[p]
            xx = np.where(y == valoare_conflict)
            p = xx[0][0]

        # Odata gasita o pozitie unde copil[p] == -1, plasam valoarea 'a'
        copil[p] = a

    # 3. Completarea restului pozitiilor libere
    # Valorile din y care nu sunt inca in copil se pun pe pozitiile ramase libere (-1)
    alele_neplasate = [y[k] for k in range(n) if not y[k] in copil]
    pozitii_libere = [k for k in range(n) if copil[k] == -1]

    copil[pozitii_libere] = alele_neplasate.copy()

    return copil

def PMX(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Functia principala de crossover PMX.
    Primeste doi parinti si returneaza doi copii prin aplicarea simetrice a operatorului.
    """
    n = len(x)

    # In acest exemplu, punctele de taiere sunt fixe (3 si 6),
    # dar in algoritmi genetici reali acestea sunt aleatorii.
    i = 3
    j = 6

    print(f"Puncte de crossover: index{i} pana la {j}")

    # Copil 1 mosteneste centrul de la x, restul prin mapare de la y
    c1 = c_PMX(x, y, i, j)
    # Copil 2 mosteneste centrul de la y, restul prin mapare de la x
    c2 = c_PMX(y, x, i, j)

    return c1, c2

if __name__ == "__main__":
    # Exemplu de test
    p1 = np.array([4, 9, 8, 3, 5, 7, 0, 1, 6, 2])
    p2 = np.array([7, 5, 1, 2, 9, 0, 3, 8, 6, 4])

    print("Parinte 1: ", p1)
    print("Parinte 2: ", p2)

    copil1, copil2 = PMX(p1, p2)

    print("Copil 1:   ",  copil1)
    print("Copil 2:   ",  copil2)