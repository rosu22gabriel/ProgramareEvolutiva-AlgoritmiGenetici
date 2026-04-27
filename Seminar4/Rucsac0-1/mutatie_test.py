import numpy as np
# Importam functia m_binar (bitflip) din biblioteca FunctiiMutatieIndivizi
from FunctiiMutatieIndivizi import m_binar
import matplotlib.pyplot as grafic
import random
import copy
from typing import List, Tuple, Union

# Definirea tipurilor pentru o mai buna structura
# Un individ binar este o lista de 0 si 1, urmata de calitatea sa (valoarea totala)
IndividBinar = List[Union[int, float]]
PopulatieBinara = List[IndividBinar]

# verifica fezabilitatea alegerii x si calculeaza si f.obiectiv
def ok(x: List[int], c: np.ndarray, v: np.ndarray, maxim: float) -> Tuple[bool, float]:
    """
    Evalueaza daca configuratia de obiecte x depaseste capacitatea rucsacului.

    Args:
        x: Vector de prezenta (1 daca obiectul e ales, 0 altfel).
        c, v: Vectori cu costurile si valorile obiectelor.
        maxim: Capacitatea rucsacului.
    """
    # np.dot calculeaza suma produselor (suma obiectelor alese)
    val = np.dot(x, v)
    cost = np.dot(x, c)
    return cost <= maxim, val

# genereaza populatia initiala
def gen(c: np.ndarray, v: np.ndarray, maxim: float, dim: int) -> PopulatieBinara:
    """
    Creeaza populatia initiala fezabila prin generare aleatorie.
    """
    n = v.size
    pop: PopulatieBinara = []
    for i in range(dim):
        gata = False
        while gata == False:
            # random.choices genereaza un individ binar aleator de lungime n
            x = random.choices([0, 1], k = n)
            gata, val = ok(x, c, v, maxim)


        # Atasam calitatea individului la finalul listei
        x.append(val)
        pop.append(x)

    # Extragem fitness-ul pentru vizualizarea initiala (puncte verzi)
    vectv = [pop[i][-1] for i in range(dim)]
    grafic.plot(vectv, "gs", markersize=11, label="Initial")
    return pop

# mutatie asupra populatiei de copii
def mutatie_populatie(pop: PopulatieBinara, c: np.ndarray, v: np.ndarray, maxim: float, pm: float) -> PopulatieBinara:
    """
    Aplica mutatia binara bitflip la nivel de gena.

    Args:
        pop: Populatie curenta.
        pm: Probabilitatea de mutatie pentru FIECARE gena (bit).
    """

    # copy.deepcopy este esential pentru a crea o copie completa a listelor imbricate
    mpop = copy.deepcopy(pop)
    dim = len(pop)
    n = c.size

    for i in range(dim):
        # Extragem genele individului (primele n elemente)
        x = pop[i][:n].copy()
        a_avut_loc_o_mutatie = False

        for j in range(n):
            # Decidem pentru fiecare gena (obiect) daca ii inversam starea
            r = random.uniform(0, 1)
            if r <= pm:
                # m_binar face: 0 -> 1 sau 1 -> 0
                x[j] = m_binar(x[j])
                a_avut_loc_o_mutatie = True

        # Daca cel putin un bit a fost schimbat, verificam daca noul individ este valid
        if a_avut_loc_o_mutatie:
            fezabil, val = ok(x, c, v, maxim)
            # Daca mutatia a dus la o solutie valida (nu depaseste greutatea), o pastram
            if fezabil:
                # Reconstruim individul cu noua calitate
                nou_individ = x + [val]
                mpop[i] = nou_individ
            # Nota: Daca mutatia nu e fezabila, individul ramane in starea de dinainte

    # Reprezentam noile calitati cu patrate rosii pentru a vedea evolutia
    grafic.plot([mpop[i][n] for i in range(dim)], "rs", markersize=7, label="Dupa mutatie")
    return mpop

if __name__ == "__main__":
    # Parametri problemei
    capacitate_max = 50
    dim_populatie = 12

    # Incarcam datele din fisiere (costurile si valorile obiectelor)
    try:
        costuri = np.genfromtxt("cost.txt")
        valori = np.genfromtxt("valoare.txt")

        # Rulam procesul de generare si mutatie
        p = gen(costuri, valori, capacitate_max, dim_populatie)

        # pm=0.1 inseamna ca fiecare bit are 10% sanse sa fie inversat
        o = mutatie_populatie(p, costuri, valori, capacitate_max, 0.1)

        grafic.title("Mutatia Bitflip in Problema Rucsacului 0-1")
        grafic.xlabel("Index Individ")
        grafic.ylabel("Calitate (Valoare)")
        grafic.legend()
        grafic.show()

    except OSError:
        print("Eroare la incarcarea fisierelor 'cost.txt' sau 'valoare.txt'.")