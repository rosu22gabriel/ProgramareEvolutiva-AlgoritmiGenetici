import copy

import numpy as np
from typing import List, Tuple, Union

# --- Enunt Cerinta (Problema 1 din PDF) ---
# Fie f : {1, 2, ..., 1500} x {-1, 0, ..., 2500} -> R,
# f(x, y) = (y-1) * (sin(x-2)) ^ functia obiectiv a unei probleme de maxim.
# Fiecarui fenotip (x, y) ii corespunde un genotip sir binar obtinut prin reprezentarea
# in baza 2 a fiecarei componente a fenotipului.
#
# a. Generarea aleatoare a unei populatii 'pop' de dimensiune 'dim'.
# b. Recombinare utilizand operatorul de incrusisare multi-punct.

def dec_to_bin(n: int, m: int) -> List[int]:
    """
    Transforma un numar intreg n intr-o lista de m binti (Reprezentare binara).

    Args:
        n: Numarul zecimal de convertit.
        m: Numarul de biti pe care se face reprezentarea.
    """
    # bin(n)[2:] scoate prefixul '0b'
    repr_str = bin(n)[2:]
    # Adaugam zerouri la stanga pana la lungimea m
    repr_f = repr_str.zfill(m)
    # Transformam string-ul in lista de intregi [0, 1, ...]
    return [int(bit) for bit in repr_f]


def bin_to_dec(x: List[int], m: int) -> int:
    """
    Transforma un vector binar de m biti inapoi intr-un numar zecimal.
    """
    y = ''.join(map(str, x))
    return int(y, 2)


def fitness(sir: List[int]) -> float:
    """
    Functia de evaluarea a calitatii unui individ (genotip).

    Interpretare genotip:
    - bitii 0-10 (11 biti): variabila x (suficienti pentru max 2047, acopera 1500)
    - bitii 11-22 (12 biti): variabila y (suficienti pentru max 4095, acopera 2501)
    """
    # Decodificam componentele fenotipice
    x_val = bin_to_dec(sir[0: 11], 11)
    # y este translatat cu -1 conform cerintei (y-1 in fitness, deci stocam y+1 in genotip)
    # sau decodificam direct si aplicam formula:
    y_val = bin_to_dec(sir[11:23], 12) - 1 # Ajustare pentru intervalul {-1, ..., 2500}

    return float(y_val * (np.sin(x_val - 2)**2))


def cerinta_a(dim: int) -> List[List[Union[int, float]]]:
    """
    Genereaza populatia initiala.
    """
    populatie = []
    print("--- Generare Populatie Initiala ---")
    for i in range(dim):
        # Generam fenotipurile conform intervalelor din enunt
        x = np.random.randint(1, 1501)
        y = np.random.randint(-1, 2501)

        # Reprezentarea binara (11 biti pentru x, 12 biti pentru y)
        # Nota: pentru y, translatam valoarea pentru a lucra cu numere pozitive in binar
        individ = dec_to_bin(x, 11) + dec_to_bin(y + 1, 12)
        calitate = fitness(individ)

        # Adaugam calitatea la sfarsitul vectorului (reprezentare cromozomiala extinsa)
        individ_complet = individ + [calitate]
        populatie.append(individ_complet)

        print(f"Individ {i}: x={x}, y={y} | Calitate={calitate:.4f}")

    return populatie


def recombinare_3puncte(sir1: List[int], sir2: List[int]) -> Tuple[List[int], List[int]]:
    """
    Operatorul de incrucisare multi-punct (3 puncte de taiere).
    """
    n = 23 # Lungimea totala a genotipului (11 + 12)

    # Generam 3 puncte de taiere distincte si sortate
    puncte = sorted(np.random.choice(range(1, n), 3, replace=False))
    i, j, k = puncte

    print(f"Puncte de taiere: {i}, {j}, {k}")

    # Construim copiii prin alternarea segmentelor intre parinti
    # Segment 1: [0:i] - P1
    # Segment 2: [i:j] - P2 (interschimbat)
    # Segment 3: [j:k] - P1
    # Segment 4: [k:n] - P2 (interschimbat)

    c1 = sir1[:i] + sir2[i:j] + sir1[j:k] + sir2[k:]
    c2 = sir2[:i] + sir1[i:j] + sir2[j:k] + sir1[k:]

    return c1, c2


def cerinta_b(populatie: List[List[Union[int, float]]], pc: float) -> List[List[Union[int, float]]]:
    """
    Realizeaza recombinarea populatiei cu probabilitatea pc.
    Include si recombinarea asexuata (copierea parintilor daca nu are loc crossover).
    """
    dim = len(populatie)
    copii = copy.deepcopy(populatie)

    print("\n--- Proces de Recombinare ---")
    for i in range(0, dim - 1, 2):
        r = np.random.uniform(0, 1)
        if r <= pc:
            # Extragem genotipurile parintilor (fara fitness-ul de pe ultima pozitie)
            p1 = populatie[i][:23]
            p2 = populatie[i+1][:23]


            # Aplicam operatorul cu 3 puncte
            c1, c2 = recombinare_3puncte(p1, p2)

            # Actualizam copiii si le calculam noul fitness
            copii[i][:23] = c1
            copii[i][23] = fitness(c1)
            copii[i+1][:23] = c2
            copii[i+1][23] = fitness(c2)
            print(f"Perechea {i}-{i+1}: Recombinare efectuata.")
        else:
            print(f"Perechea {i}-{i+1}: Recombinare asexuata (copiere).")
    return copii

if __name__ == "__main__":
    # Parametri
    DIM_POPULATIE = 10
    PROB_CROSSOVER = 0.8

    # Rulare pasi ceruti
    pop_init = cerinta_a(DIM_POPULATIE)
    pop_copii = cerinta_b(pop_init, PROB_CROSSOVER)

    print("\n --- Populatia Rezultata (Copii) ---")
    for idx, ind in enumerate(pop_copii):
        print(f"Individ {idx}: Fitness {ind[23]:.4f}")
