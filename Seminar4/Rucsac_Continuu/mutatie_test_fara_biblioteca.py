import numpy as np
import matplotlib.pyplot as grafic
import random
import copy
from typing import List, Tuple, Union

# Definirea tipurilor pentru claritate
# Un individ este o lista de fractiuni (float) plus fitness-ul la final
Individ = List[float]
Populatie = List[Individ]

# verifica fezabilitatea alegerii si calculeaza si f. obiectiv
def ok(x: List[float], c: np.ndarray, v: np.ndarray, maxim: float) -> Tuple[bool, float]:
    """
    Verifica daca rucsacul este valid si calculeaza valoare fitness.

    Args:
        x: Genele individului (fractiuni intre 0 si 1)
        c, v: Vectorii de cost si valoare
        maxim: Capacitatea rucsacului.
    """
    val = np.dot(x, v)
    cost = np.dot(x, c)
    return cost <= maxim, val


# genereaza populatia initiala
def gen(c: np.ndarray, v: np.ndarray, maxim: float, dim: int) -> Populatie:
    """
    Creeaza populatia initiala prin esantionare continua.
    """
    n = v.size
    pop: Populatie = []
    for i in range(dim):
        gata = False
        while gata == False:
            # Generam genele x ca numere reale intre 0 si 1
            x = [random.uniform(0, 1) for _ in range(n)]
            gata, val = ok(x, c, v, maxim)

        # Adaugam valoarea fitness ca ultim element in lista
        x.append(val)
        pop.append(x)

    # Extragem doar coloana de fitness pentru graficul initial
    vectv = [pop[i][-1] for i in range(dim)]
    grafic.plot(vectv, "gs", markersize=11, label="Initial")
    return pop

# mutatie neuniforma pentru numarul real x din (a, b)
def mutatie_neuniforma(x: float, sigma: float, a: float, b: float) -> float:
    """
    Perturba o valoare reala folosind o distributie normala (Gauss).

    Args:
        x: Valoarea actuala a genei
        sigma: Pasul de mutatie (abatarea standard). Controleaza cat de mare e "socul".
        a, b: Limitele intervalului (aici 0 si 1).
    """
    # random.gauss(media, deviatia_standard)
    perturbare = random.gauss(0, sigma)
    y = x + perturbare

    # Mentinem valoarea in intervalul permis [a, b]
    if y < a:
        y = a
    elif y > b:
        y = b
    return y


# aplica mutatie populatiei copii
def mutatie_populatie(copii: Populatie, c: np.ndarray, v: np.ndarray, cmax: float, pm: float, sigma: float) -> Populatie:
    """
    Aplica mutatia neuniforma la nivel de gena pentru fiecare individ.

    Args:
        copii: Populatia de reprodus.
        pm: Probabilitatea de mutatie pentru FIECARE gena in parte.
        sigma: Intensitatea mutatiei.
    """
    # Folosim deepcopy pentru a nu modifica populatia originala (parintii)
    copii_m = copy.deepcopy(copii)
    dimensiune = len(copii)
    n = c.size

    for i in range(dimensiune):
        # x primeste doar genele (fara fitness)
        x = copii_m[i][:-1].copy()
        mutatie_efectuata = False

        for j in range(n):
            # Decidem pentru fiecare gena daca o mutam
            raspuns = random.uniform(0, 1)
            if raspuns <= pm:
                x[j] = mutatie_neuniforma(x[j], sigma, 0, 1)
                mutatie_efectuata = True

        # Daca individul a suferit cel putin o mutatie, ii verificam noua stare
        if mutatie_efectuata:
            fezabil, val = ok(x, c, v, cmax)
            # Strategie de conservare: pastram mutatia doar daca rezultatul este valid
            if fezabil:
                x.append(val)
                copii_m[i] = x.copy()
            # Nota: Daca nu e fezabil, individul ramane in varianta lui nemutata

    # Adaugam noile calitati pe grafic cu rosu
    calitati_noi = [copii_m[i][n] for i in range(dimensiune)]
    grafic.plot(calitati_noi, "rs", markersize=7, label="Dupa mutatie")
    return copii_m


if __name__ == "__main__":
    # Parametri problemei
    capacitate_maxima = 50
    dimensiune_populatie = 12

    # Incarcare date
    try:
        costuri = np.genfromtxt("cost.txt")
        valori = np.genfromtxt("valoare.txt")

        # Generare si Mutatie
        p = gen(costuri, valori, capacitate_maxima, dimensiune_populatie)
        # pm = 0.1 (10% sanse de mutatie per gena), sigma = 0.7 (perturbare medie spre mare)
        o = mutatie_populatie(p, costuri, valori, capacitate_maxima, 0.1, 0.7)

        grafic.title("Evolutia Fitness-ului: Mutatie Neuniforma (Real-coded GA)")
        grafic.xlabel("Index Individ")
        grafic.ylabel("Valoare Rucsac")
        grafic.legend()
        grafic.show()
    except OSError:
        print("Fisierele de date lipsesc.")