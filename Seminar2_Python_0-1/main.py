from typing import Tuple

import numpy as np

def calculeaza_functie(cost: np.ndarray, valoare: np.ndarray, x: np.ndarray, Cmax: float) -> Tuple[bool, float]:
    """
    Evalueaza daca o solutie respecta restrictia de capacitate si ii calculeaza valoare totala.

    Args:
        cost: Vectorul costurilor obiectelor.
        valoare: Vectorul valorilor obiectelor.
        x: vectorul binar care reprezinta solutia curent (1 = obiect ales, 0 = nu).
        Cmax: Capacitatea maxima admisa (restrictia).

    Returns:
        Un tuplu (respecta_restrictia, valoare_totala).
    """
    cost_x = np.dot(x, cost)
    valoare_x = np.dot(x, valoare)
    # Verificam daca solutia este valida (nu depaseste Cmax)
    return cost_x <= Cmax, valoare_x

def calculeaza_vecini(cost: np.ndarray, valoare: np.ndarray, x: np.ndarray, Cmax: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genereaza toti vecini posibili ai unei solutii prin inversarea fiecarui bit (Hamming instance = 1).
    Pastreaza doar vecinii care respecta restrictia de capacitate.
    """
    vecini_x = []
    valoare_vecini_x = []

    n = cost.size

    for i in range(n):
        # Cream o copie a solutiei curente si inversam bitul de pe pozitia i
        y = x.copy()
        y[i] = 1 - x[i] # Inversare binara (daca e 1 devine 0, daca e 0 devine 1)

        este_valid, valoare_y = calculeaza_functie(cost, valoare, y, Cmax)
        if este_valid:
            vecini_x.append(y)
            valoare_vecini_x.append(valoare_y)

    return np.array(vecini_x), np.array(valoare_vecini_x)

def HC(fc: str, fv: str, iteratii: int, Cmax: float) -> None:
    """
    Algoritmul Hill Climbing (Random Restart Hill Climbing).

    Args:
        fc: Calea catre fisierul cu costuri.
        fv: Calea catre fisierul cu costuri.
        iteratii: De cate ori repornim algoritmul din puncte aleatorii (Restarts).
        Cmax: Capacitatea rucsacului.
    """
    # Incarcarea datelor din fisiere
    cost = np.genfromtxt(fc)
    valoare = np.genfromtxt(fv)
    n = cost.size

    # Structuri pentru a retine rezultatul fiecarei reporniri (restart)
    valoare_iteratii = np.zeros(iteratii, float)
    best_iteratii = np.zeros([iteratii, n], int)

    for i in range(iteratii):
        gasit = False
        local = False

        # 1. Generare punct de start: Cautam o solutie initiala valida (random)
        while not gasit:
            x = np.random.randint(0, 2, n)
            gasit, val = calculeaza_functie(cost, valoare, x, Cmax)

        # 2. Urcarea pe munte (Hill Climbing propriu_zis)
        while not local:
            vecini_x, valoare_vecini_x = calculeaza_vecini(cost, valoare, x, Cmax)

            if valoare_vecini_x.size == 0:
                # Daca nu avem vecini valizi, suntem intr-un punct blocat
                local = True
            else:
                # Cautam cel mai bun vecin (Best Improvement)
                j = np.argmax(valoare_vecini_x)
                val_max_vecini = valoare_vecini_x[j]

                if val_max_vecini > val:
                    # Daca am gasit un vecin mai bun, ne mutam acolo
                    val = val_max_vecini
                    x = vecini_x[j]
                else:
                    # Daca niciun vecin nu e mai bun decat punctul curent, am atins un optim local
                    local = True

        # Salvam cel mai bun punct gasit in aceasta iteratie (restart)
        valoare_iteratii[i] = val
        best_iteratii[i] = x

    # 3. Rezultat final: Alegem cea mai buna solutie dintre toate restartarile
    index_optim = np.argmax(valoare_iteratii)
    best_global = best_iteratii[index_optim]
    valoare_best_global = valoare_iteratii[index_optim]

    print("-" * 30)
    print(f"REZULTAT FINAL:")
    print(f"Cea mai buna valoare: {valoare_best_global}")
    print(f"Configuratia (x): {best_global}")
    print("-" * 30)

if __name__ == "__main__":
    try:
        HC("cost.txt", "valoare.txt", 70, 50)
    except FileNotFoundError:
        print("Eroare: Fisierele cu date nu au fost gasite.")