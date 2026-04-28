import numpy as np
import matplotlib.pyplot as grafic
from typing import List, Union

# --- Enunt Cerinta ---
# Fie f: Pn -> N, p apartine Pn, f(p) = |{(i, j) | i < j, p[i] < p[j]| functia obiectiv a unei probleme de maxim.
# Pn reprezinta multimea permutarilor de n elemente.
#
# a) Scrieti o functie pentru generarea aleatoare a unei populatii de dimensiune dim;
#    calitatea fiecarui individ este memorata la sfarsitul fiecarei reprezentari.
# b) Pentru o probabilitate de mutatie data, pm, scrieti o functie de mutatie utilizand operatorul
#    de mutatie prin amestec (scramble mutation) care obtine o noua populatie popm.
# ----------------------

def fitness(p: np.ndarray) -> int:
    """
    Calculeaza valoarea functiei obiectiv f(p).
    Numara perechile (i, j) care respecta ordinea crescatoare atat la indici cat si la valori.

    Args:
        p: Vector numpy reprezentand o permutare.
    Returns:
        Valoare fitness (numarul de perechi p[i] < p[j] cu i < j).
    """
    n = p.size
    val = len([(i, j) for i in range(n - 1) for j in range(i + 1, n) if p[i] < p[j]])
    return val

def cerinta_a(dim: int, n: int) -> np.ndarray:
    """
    Genereaza populatia initiala formata din permutari aleatorii.

    Args:
        dim: Numarul de indivizi (rute/permutari).
        n: Numarul de elemente din permutare.
    Returns:
        Matrice [dim x n+1] cu fitness pe ultima coloana.
    """
    # Alocam spatiu pentru n gene + 1 coloana de fitness
    populatie = np.zeros([dim, n + 1], dtype="int")

    for i in range(dim):
        # Fiecare rand primeste o permutare unica a numerelor 0...n-1
        populatie[i, :n] = np.random.permutation(n)
        # Calculam calitatea (fitness)
        populatie[i, n] = fitness(populatie[i, :n])

    # Reprezentare grafica: patrate albastre (bs)
    grafic.plot([populatie[i][-1] for i in range(dim)], "bs", markersize=12, label="Populatie initiala")
    return populatie

def amestec(p: np.ndarray) -> np.ndarray:
    """
    Realizeaza mutatia prin amestec (Scramble Mutation).
    Selecteaza o subsecventa si rearanjeaza elementele acesteia in mod aleatoriu.

    Args:
        p: Individul (permutarea) originala.
    Returns:
        Noul individ dupa amestecarea secventei.
    """
    n = len(p)
    # Alegem doua puncte de taiere i si j
    i = np.random.randint(0, n - 1)
    j = np.random.randint(i + 1, n)

    r = p.copy()
    # Extragem secventa dintre cele doua puncte
    secventa = r[i:j + 1].copy()

    # Reordonam aleatoriu elementele din secventa (fara inlocuire)
    secventa_noua = np.random.choice(secventa, secventa.size, replace=False)

    # Punem secventa amestecata inapoi in individ
    r[i:j + 1] = secventa_noua.copy()

    print(f"\nAmestec (Scramble) in permutarea {p}, intre pozitiile ({i}, {j})")
    print(f"Rezultat: {r}")
    return r

def cerinta_b(populatie: np.ndarray, pm: float) -> np.ndarray:
    """
    Aplica mutatia prin amestec la nivelul intregii populatii.

    Args:
        populatie: Matricea populatiei actuale.
        pm: Probabilitatea de mutatie (0.0 - 1.0).
    Returns:
        Noua populatie rezultata (popm).
    """
    dim = populatie.shape[0]
    populatie_m = populatie.copy()

    for i in range(dim):
        # Decidem daca individul i sufera o mutatie
        r = np.random.uniform(0, 1)
        if r <= pm:
            # Genele sunt pe toate coloanele exceptand ultima
            populatie_m[i, :-1] = amestec(populatie[i, :-1])
            # Recalculam fitness pentru noul individ
            populatie_m[i, -1] = fitness(populatie_m[i, :-1])

    # Reprezentare grafica: patrate rosii (rs)
    grafic.plot([populatie_m[i][-1] for i in range(dim)], "rs", markersize=9, label="Populatia dupa mutatie")
    return populatie_m

if __name__ == "__main__":
    # Parametri test
    n_elemente = 10
    dimensiune_pop = 12
    prob_mutatie = 0.4 # Probabilitate ridicata pentru a observa mutatiile in consola

    # Executie Cerinta A
    pop_init = cerinta_a(dimensiune_pop, n_elemente)
    print("Populatia initiala: ")
    print(pop_init)

    # Executie Cerinta B
    pop_final = cerinta_b(pop_init, prob_mutatie)
    print("\n Populatia dupa mutatie:")
    print(pop_final)

    # Finalizare grafic
    grafic.title("Mutatia prin Amestec (Scramble Mutation)")
    grafic.xlabel("Index Individ")
    grafic.ylabel("Fitness (f(p))")
    grafic.legend()
    grafic.show()