import numpy as np

# =============================================================================
# OPERATORUL PMX (Partially Mapped Crossover)
# =============================================================================

def c_PMX(x: np.ndarray, y: np.ndarray, p1: int, p2: int) -> np.ndarray:
    """
    Construieste un singur copil dintr-o pereche de parinti folosind mecanismul PMX.

    Args:
        x: Primul parinte (furnizeaza secventa principala).
        y: Al doilea parinte (furnizeaza restul genelor si maparea).
        p1, p2: Punctele de taiere (secventa dintre ele va fi mostenita direct).
    """
    n = x.size
    # Initializam copilul cu -1 pentru a marca pozitiile libere
    copil = -np.ones(n, dtype="int")

    # PAS 1: Copiem secventa cuprinsa intre p1 si p2 din primul parinte x
    copil[p1:p2 + 1] = x[p1:p2+1]

    # PAS 2: Tratam conflictele pentru genele din secventa (p1, p2) a parintelui y
    # Cautam acele gene din y[p1:p2+1] care NU se regasesc deja in secventa luata din x
    poz_alele_neplasate = [k for k in range(p1, p2 + 1) if not y[k] in copil]

    for p in poz_alele_neplasate:
        val_y = y[p]
        curr_p = p

        # Cautam o pozitie libera in afara secventei [p1, p2] folosind maparea
        # Daca pozitia la care vrem sa punem gena este ocupata, "sarim" la indexul
        # unde se afla acea valoare in parintele y, pana gasim un loc liber (-1)
        while copil[curr_p] > -1:
            val_conflict = copil[curr_p]
            # Gasim unde se afla valoarea care ne blocheaza in parintele y
            idx_in_y = np.where(y == val_conflict)[0][0]
            curr_p = idx_in_y

        copil[curr_p] = val_y

    # PAS 3: Completam restul pozitiilor ramase libere
    # Luam din y genele care nu au fost inca plasate in copil
    alele_neplasate = [y[k] for k in range(n) if not y[k] in copil]
    pozitii_libere = [k for k in range(n) if copil[k] == -1]

    # Plasam genele ramase in ordinea in care apar in parintele y
    copil[pozitii_libere] = alele_neplasate
    return copil

def PMX(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Functie wrapper care aplica mecanismul c_PMX de doua ori pentru a genera doi copii.
    """
    n = len(x)
    # In acest exemplu, punctele de taiere sunt fixe (3 si 6) pentru demonstratie
    i = 3
    j = 6

    print(f"Puncte de taiere: {i}, {j}")

    # Generam primul copil (secventa din x, mapare din y)
    c1 = c_PMX(x, y, i, j)
    # Generam al doilea copil (secventa din y, mapare din x)
    c2 = c_PMX(y, x, i, j)

    return c1, c2


# =============================================================================
# TESTARE OPERATOR
# =============================================================================

if __name__ == "__main__":
    # Definirea a doi părinți ca permutări de 10 elemente
    x = np.array([4, 9, 8, 3, 5, 7, 0, 1, 6, 2])
    y = np.array([7, 5, 1, 2, 9, 0, 3, 8, 6, 4])

    print("Părinte 1: ", x)
    print("Părinte 2: ", y)

    c1, c2 = PMX(x, y)

    print("Copil 1:   ", c1)
    print("Copil 2:   ", c2)