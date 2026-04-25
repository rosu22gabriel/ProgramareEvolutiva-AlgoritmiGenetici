import numpy as np
from math import sin, cos
import matplotlib.pyplot as grafic
from typing import List, Tuple, Union

# Definirea tipurilor pentru claritate
Vector = Union[List[float], np.ndarray]

def f_obiectiv(x: float) -> float:
    # o functie de maximizat cu mai multe puncte de maxim local

    # I: x - punctul in care se calculeaza valoarea functiei
    # E: y - valoarea functiei in punctul x

    # o functie oarecare cu extreme locale

    # x**3 - x la puterea a 3 a
    y = x**3 * sin(x/3) + x**3 * cos(2*x) - 2*x*sin(3*x) + 2*x*cos(x)

    return y

def vecini(x: float, nr: int, pas: float, a: float, b: float) -> List[List[float]]:
    # calculeaza vecinii unui punct (pe o axa), impreuna cu calitatile acestora

    # I: x - punct curent
    #    nr - numar de vecini pe fiecare directie
    #    pas - distanta intre vecinii consecutivi
    #    a, b - capetele intervalului de lucru
    # E: v - lista vecini (2 linii: puncte, calitati)

    # sunt folosite liste comprehensive
    # vec: genereaza puncte la stanga si la dreapta lui x, verificand sa ramana in [a, b]
    vec = [ x + i * pas for i in range(-nr, nr + 1) if ((x + i * pas >= a) and (x + i * pas <= b))]
    # valv: calculeaza valoarea functiei (fitness-ul) pentru fiecare punct generat mai sus
    valv = [f_obiectiv(x + i * pas) for i in range(-nr, nr + 1) if ((x + i * pas >= a) and (x + i * pas <= b))]

    return [vec, valv]


def HC(a: float, b: float, nrp: int, nrv: int, pas: float) -> Tuple[float, float]:
    # implementare hillclimbing pentru gasirea maximului unei functii de o variabila

    # I: a, b - capetele intervalului pe care e definita functia (o axa)
    #    nrp - numarul de puncte initiale folosite de algoritm (Random Restarts)
    #    nrv - numarul de vecini pe fiecare directie utilizatori (total 2*nrv+1 - 1 e punctul curent)
    #    pas - distanta intre doi vecini consecutivi
    # E: x - punct de maxim
    #    fx - valoarea maxima a functiei (in punctul x)
    # Obs: Daca nu se inchide dupa rularea unui exemplu, la rularea urmatorului exemplu
    # graficul va fi adaugat peste cel anterior

    # initializare liste goale de coordonate pentru a stoca rezultatul fiecarei rulari
    X: List[Union[None, float]] = [None] * nrp
    Y: List[Union[None, float]] = [None] * nrp

    # pentru fiecare punct initial (fiecare restart al algoritmului)
    for i in range(nrp):
        # aplicare hillclimbing pentru punctul initial curent generat aleator
        pc = np.random.uniform(a, b)    # alege un punct de inceput aleator (Explorare)
        local = 0                       # flag pentru a marca daca am stagnat intr-un optim local

        while not local:
            # Calculeaza vecinii punctului curent si valorile corespunzatoare ale functiei
            nvec, nval = vecini(pc, nrv, pas, a, b)

            # Gaseste cel mai bun vecin din lista (Cautare de tip "Best Improvement")
            valmax = max(nval)
            poz = nval.index(valmax)
            vecmax = nvec[poz]

            # inlocuieste punctul curent cu cel mai bun vecin, daca exista unul mai bun (Exploatare)
            if valmax > f_obiectiv(pc):
                pc = vecmax
            else:
                # nici un vecin mai bun, inseamna ca am atins varful dealului (
                local = 1

        X[i] = vecmax
        Y[i] = f_obiectiv(vecmax)

    # determina cel mai bun dintre toate punctele finale colectate din cele nrp restartari
    fx = max(Y) # type: ignore
    poz = Y.index(fx)
    x = X[poz] # type: ignore

    # afiseaza rezultatele si apeleaza functia de vizualizare
    print("Valoare maxima calculata: ", fx)
    print("E atinsa in punctul: ", x)
    deseneaza(a, b, X, Y, x, fx) # type: ignore

    return x, fx


def deseneaza(a: float, b: float, X: List[float], Y: List[float], xmax: float, ymax: float) -> None:
    # vizualizare rezultate pentru hillclimbing 1 variabila

    # I: a, b - capete interval de lucru
    #    X, Y - Liste cu coordonatele punctelor finale calculate (toate optimele locale gasite)
    #    xmax, ymax - coordonatele celui mai bun punct gasit (optimul global estimat)

    # Generam puncte pentru a desena curba continua a functiei
    # .arange genereaza un tablou (array) de numere egal distantate intr-un interval specificat
    # intoarce np.ndarray, permite utilizare pasului zecimal (0.1)
    # sintaxa: np.arange([start], stop, [step], dtype=None)
    x = np.arange(a, b, 0.01)

    # Desenam functia obiectiv cu negru (k-)
    grafic.plot(x, [f_obiectiv(i) for i in x], 'k-', alpha=0.7)

    # Marcam cu puncte albastre (bo) toate punctele unde algoritmul s-a oprit
    grafic.plot(X, Y, 'bo', label='optim local', markersize=4)

    # Marcam cu o steluta rosie (r*) cel mai inalt punct gasite dintre toate
    grafic.plot(xmax, ymax, 'r*', label='cel mai bun calculat', markersize=10)

    grafic.title("Optimizare Hill Climbing cu Random Restart")
    grafic.xlabel("x")
    grafic.ylabel("f(x)")
    grafic.legend()
    grafic.grid(True, linestyle="--", alpha=0.6)
    grafic.show()


if __name__ == "__main__":
    # Parametri: interval [0, 50], 100 restartari, 100 vecini analizati per pas, pas de 0.01
    x_opt, fx_opt = HC(0, 50, 100, 100, 0.01)