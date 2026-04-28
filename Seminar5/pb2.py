import numpy as np
import matplotlib.pyplot as grafic
from typing import List, Union

# --- Enunt Cerinta ---
# Fie f: [-1, 1] x [0, 0.2] x [0, 1] x [0, 5] -> R,
# f(x1, x2, x3, x4) = 1 + sin(2*x1 - x3) + (x2 * x4) ^ (1/3)
# functia obiectiv a unei probleme de maxim. Un genotip este un vector x = [x1, x2, x3, x4]
#
# a) Scrieti o functie pentru generarea aleatoare a unei populatii de dimensiune dim;
#    calitatea este memorata la sfarsitul fiecarei reprezentari.
# b) Pentru o probabilitate pm, scrieti o functie de mutatie de tip neuniform
#    cu pragul t=0.6 si sigma = t/3, care sa obtina o noua populatie.
# ----------------------

def fitness(x: np.ndarray) -> float:
    """
    Calculeaza valoarea functiei obiectiv pentru un individ cu gene reale.

    Args:
        x: Vector cu 4 componente (x1, x2, x3, x4).
    Returns:
        Valoarea fitness conform formulei sinusoide.
    """
    # Formula: 1 + sin(2*x1 - x3) + radacina_cubica(x2 * x4)
    return 1 + np.sin(2 * x[0] - x[2]) + (x[1] * x[3]) ** (1/3)

def cerinta_a(dim: int, a: List[float], b: List[float]) -> np.ndarray:
    """
    Genereaza populatia initiala in spatiul continuu (numere reale).

    Args:
         dim: Dimensiunea Populatiei.
         a: Lista limitelor inferioare pentru cele 4 variabile.
         b: Lista limitelor superioare pentru cele 4 variabile.

    Returns:
        Matrice [dim x 5] (4 gene + 1 fitness)
    """
    populatie = np.zeros([dim, 5])
    for i in range(dim):
        # Generam 4 gene reale in intervalele specifice [a_j, b_j]
        populatie[i, :4] = np.random.uniform(a, b, 4)
        # Calculam calitatea individului
        populatie[i, 4] = fitness(populatie[i, :4])

    # Plotare fitness populatie initiala cu patrate albastre (bs)
    grafic.plot([populatie[i][-1] for i in range(dim)], "bs", markersize=12, label="Populatia initiala")
    return populatie

def mutatie_neuniforma(x: float, a: float, b: float, sigma: float) -> float:
    """
    Realizeaza mutatia unei singure gene folosind o distributie normala (Gaussiana).

    Args:
        x: Valoarea actuala a genei.
        a: Limita inferioara a domeniului genei.
        b: Limita superioara a domeniului genei.
        sigma: Deviatia standard pentru perturbare.
    Returns:
        Noua valoare a genei, incadrata in intervalul [a, b].
    """
    # Generam o perturbare bazata pe distributia normala N(0, sigma)
    perturbare = np.random.normal(0, sigma)
    y = x + perturbare

    # Verificam respectarea frontierelor domeniului de definitie
    if y < a:
        y = a
    elif y > b:
        y = b
    return y

def cerinta_b(populatie: np.ndarray, pm: float, a: List[float], b: List[float], sigma: float) -> np.ndarray:
    """
    Aplica mutatia neuniforma asupra fiecarei gene a fiecarui individ din populatie.

    Args:
        populatie: Matricea populatiei curente.
        pm: Probabilitatea de mutatie la nivel de gena.
        a: Limitele inferioare.
        b: Limitele superioare.
        sigma: Parametrul de dispersie.
    """
    dim = populatie.shape[0]
    populatie_m = populatie.copy()

    for i in range(dim):
        mutatie_efectuata = False
        # Parcurgem fiecare dintre cele 4 gene ale individului
        for j in range(4):
            r = np.random.uniform(0, 1)
            if r <= pm:
                # Aplicam mutatia pe gena j
                populatie_m[i, j] = mutatie_neuniforma(populatie[i, j], a[j], b[j], sigma)
                mutatie_efectuata = True

        # Daca individul a suferit modificari, ii recalculam fitness-ul
        if mutatie_efectuata:
            populatie_m[i, -1] = fitness(populatie_m[i, : -1])

            # Afisare detalii pentru monitorizare
            original = [float(f"{val:.4f}") for val in populatie[i][:-1]]
            mutat = [float(f"{val:.4f}") for val in populatie_m[i][:-1]]
            print(f"\nMutatie in individul {original} (fit: {populatie[i][-1]:.3f})")
            print(f"Rezulta individul {mutat} (fit: {populatie_m[i][-1]:.3f}")

    # Plotare fitness populatie mutanta cu patrate rosii (rs)
    grafic.plot([populatie_m[i][-1] for i in range(dim)], "rs", markersize=9, label="Populatia dupa mutatie")
    return populatie_m

if __name__ == "__main__":
    # Configurare experiment conform cerintei
    dimensiune = 12
    prob_mutatie = 0.1 #Probabilitate per gena

    # Domeniile de definitie pentru [x1, x2, x3, x4]
    limite_inf = [-1, 0, 0, 0]
    limite_sup = [1, 0.2, 1, 5]

    # Parametru sigma calculat conform regulii t/3 (t=0.6)
    sigma_param = 0.6 / 3 # Aproximativ 0.2

    # Rulare proceduri
    pop_init = cerinta_a(dimensiune, limite_inf, limite_sup)
    pop_final = cerinta_b(pop_init, prob_mutatie, limite_inf, limite_sup, sigma_param)

    # Finalizare grafic
    grafic.title("Algoritm Genetic: Mutatie Neuniforma (Reale)")
    grafic.xlabel("Individ")
    grafic.ylabel("Fitness")
    grafic.legend()
    grafic.show()