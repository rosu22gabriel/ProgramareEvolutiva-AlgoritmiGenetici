import numpy as np
from typing import Tuple, List, Union

# Definirea tipurilor pentru claritate in ndarray
# ndarray poate contine intregi sau float-uri in functie de context
VectorInt = np.ndarray
VectorFloat = np.ndarray
MatrixInt = np.ndarray

def ok(x: VectorInt, c: VectorFloat, v: VectorFloat, maxim: float) -> Tuple[bool, float]:
    """
    Verifica daca solutia x este fezabila (nu depaseste capacitatea) si calculeaza fitness-ul.

    Args:
        x: Vector binar (0/1) reprezentand obiectele alese.
        c: Vectorul costurilor (greutatilor).
        v: Vectorul valorilor.
        maxim: Capacitatea maxima a rucsacului

        Returns:
            Un tuplu (este_fezabil, valoare_totala).
    """
    # np.dot realizeaza produsul scalar: suma(x[i] * c[i])
    cost = np.dot(x, c)
    val = np.dot(x, v)
    return cost <= maxim, val

def vecini(x: VectorInt, c: VectorFloat, v: VectorFloat, maxim: float) -> Tuple[MatrixInt, VectorFloat]:
    """
        Genereaza toti vecinii valizi ai punctului x prin inversarea fiecarui bit (bit-flip).
    """
    n = x.size
    # Initializam vectori goi pentru a stoca vecinii si calitatile lor
    Vec = np.zeros(0, dtype="int")
    Cal = np.zeros(0, dtype="float")

    for i in range(n):
        y = x.copy()
        # Bit-flip pe pozitia i: daca e 1 devine 0, daca e 0 devine 1 (not x[i])
        y[i] = not x[i]

        fez, val = ok(y, c, v, maxim)
        if fez:
            # np.append adauga elementele la sfarsitul vectorului plat
            Vec = np.append(Vec, y)
            Cal = np.append(Cal, val)

    # Deoarece Vec este un vector plat (1D), trebuie sa il reformulam in matrice (2D)
    # Fiecare linie va reprezenta un vecin de lungime n
    dim = len(Vec)
    if dim > 0:
        Vec = Vec.reshape(round(dim / n), n)

    return Vec, Cal

def HC(fc: str, fv: str, dim: int, maxim: float) -> Tuple[VectorInt, float, MatrixInt, VectorFloat]:
    """
    Algoritmul Hill Climbing cu reporniri aleatoare (Random Restart).

    Args:
        fc: Fisierul cu costuri.
        fv: Fisierul cu valori.
        dim: Numarul de puncte de start (iteratii/restartari).
        maxim: Capacitatea maxima a rucsacului.
    """

    # Citirea datelor din fisierele text
    c = np.genfromtxt(fc)
    v = np.genfromtxt(fv)
    n = c.size

    # Matrice pentru a stoca cel mai bun punct din fiecare restart
    puncte = np.zeros([dim, n], dtype="int")
    # Vector pentru a stoca valoarea corespunzatoare fiecarui punct optim local
    calitati = np.zeros(dim, dtype="float")

    for apeluri in range(dim):
        # Generarea punctului initial
        local = False
        gata = False

        # Cautam un punct de start aleator care sa fie valid (sa incapa in rucsac)
        while gata == False:
            x = np.random.randint(0, 2, n)
            gata, val = ok(x, c, v, maxim)

        # Incepe urcarea (optimizarea locala)
        while not local:
            Vec, Cal = vecini(x, c, v, maxim)

            # Daca nu exista niciun vecin valid (rucsacul e plin sau nicio schimbare nu e ok)
            if Cal.size == 0:
                local = True
            else:
                # Alegem cel mai bun vecin din lista (Criteriul "Best Improvement")
                i = np.argmax(Cal)
                vn = Vec[i]

                # Daca cel mai bun vecin este mai bun decat punctul curent, ne mutam acolo
                if Cal.max() > val:
                    val = Cal.max()
                    x = vn
                else:
                    # Am ajuns intr-un optim local (nicio mutare la vecini nu ne mai imbunatateste)
                    local = True

        # Salvam rezultatul gasit la aceasta restartare
        puncte[apeluri] = x.copy()
        calitati[apeluri] = val

    # Dupa toate restartarile, extragem cea mai buna solutie globala gasita
    vmax = np.max(calitati)
    idx_optim = np.argmax(calitati)
    sol = puncte[idx_optim]

    print("-" * 40)
    print(f"Fisiere procesate: {fc}, {fv}")
    print("Cea mai buna valoare calculata: ", vmax)
    print("Alegerea corespunzatoare este: ", sol)

    return sol, vmax, puncte, calitati

# Blocul principal de executie
if __name__ == "__main__":
    # Testarea algoritmului pe seturile de date mentionate
    try:
        # Primul test: capacitate 50, 70 restartari
        sol, val, P, C = HC("cost.txt", "valoare.txt", 70, 50)

        # Al doilea test: capacitate 50, 90 restartari
        sol1, val1, P1, C1 = HC("cost1.txt", "valoare1.txt", 90, 50)

        # Al treilea test: capacitate 56.6, 1000 de restartari
        sol2, val2, P2, C2 = HC("cost2.txt", "valoare2.txt", 1000, 56.6)

    except OSError:
        print("Atentie: Fisierele .txt necesare nu au fost gasite in directorul curnet. ")