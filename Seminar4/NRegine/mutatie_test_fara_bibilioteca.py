import numpy as np
import matplotlib.pyplot as grafic
from typing import Tuple

# f. obiectiv
def foNR(x: np.ndarray) -> int:
    """
    Calculeaza calitatea unui individ in problema celor N-Regine

    Args:
        x: Vector NumPy reprezentand permutarea randurilor
    Returns:
        c: Numarul de perechi de regine care NU se ataca pe diagonala

    """
    # Foloseste list comprehension pentru a verifica toate combinatiile i, j.
    # si numara perechile unde conditia de atac pe diagonala este falsa.
    n = x.size
    c = len([(i, j) for i in range(n - 1) for j in range(i + 1, n) if abs(i - j) != abs(x[i] - x[j])])

    return c

# genereaza populatia initiala
def gen(n: int, dim: int) -> np.ndarray:
    """
    Creeaza populatia initiala si o reprezinta grafic cu verde.

    Args:
        n: Dimensiunea tablei/numarul de regine.
        dim: Numarul de indivizi din populatie.
    """

    # Matricea populatiei: dim randuri, n gene + 1 coloana pentru calitate
    pop = np.zeros((dim, n + 1), dtype="int")
    for i in range(dim):
        # Generam o permutare aleatorie
        pop[i, :n] = np.random.permutation(n)
        # Calculam fitness-ul si il salvam pe ultima coloana
        pop[i, n] = foNR(pop[i, :n])

    # Marcam calitatile initiale cu cercuri verzi ("go")
    grafic.plot(pop[:, n], "go", markersize=11, label="Initial")
    return pop

# mutatia prin inserare in permutarea p
def mutatie_inserare(p: np.ndarray) -> np.ndarray:
    """
    Realizeaza mutatia unui singur individ prin metoda inserarii.
    Alege un element la intamplare (de la pozitia p2) si il insereaza dupa o alta pozitie (p1).
    """
    n = p.size
    print(f"\n\nmutatie in {p} cu calitatea {foNR(p)}")

    # Alegem doua pozitii aleatorii p1 si p2
    p1 = np.random.randint(0, n - 1)
    p2 = np.random.randint(p1 + 1, n)

    print(f"\npozitiile sunt {p1 + 1} si {p2 + 2}")

    rez = p.copy()
    # Logica de inserare:
    if p1 + 1 < p2:
        element_inserat = p[p2]
        # Stergem elementul de pe pozitia p2
        x = np.delete(p, p2)
        # Il inseram imediat dupa pozitia p1
        rez = np.insert(x, p1 + 1, element_inserat)

    print(f"\nrezulta individul {rez} cu calitatea {foNR(rez)}")
    return rez

# mutatie asupra populatiei de copii
def mutatie_populatie(pop: np.ndarray, dim: int, n: int, pm: float) -> np.ndarray:
    """
    Aplica operatorul de mutatie asupra intregii populatii pe baza unei probabilitati

    Args:
        pop: Matricea populatiei actuale.
        dim: Numarul de indivizi.
        n: Numarul de gene.
        pm: Probabilitatea de mutatie (uzual intre 0.01 si 0.1).
    """
    mpop = pop.copy()
    for i in range(dim):
        # Generam un numar aleator r inte 0 si 1
        r = np.random.uniform(0, 1)

        # Daca r <= pm, individul i sufera o mutatie
        if r <= pm:
            # Extragem doar genele pentru mutatie ( fara coloana de calitate)
            x = mutatie_inserare(mpop[i, :n])
            # Actualizam genele in noua populatie
            mpop[i, :n] = x.copy()
            # Recalculam calitatea pentru noul individ mutat
            mpop[i, n] = foNR(x)

    # Marcam noile calitati cu puncte rosii ("ro") pentru a observa schimbarea
    grafic.plot(mpop[:,n], "ro", markersize=7, label="Dupa mutatie")
    return mpop

if __name__ == "__main__":
    # Initializam o populatie de 18 indivizi pentru tabala de 12x12
    p = gen(12, 18)

    # Aplicam mutatia cu o probabilitate de 20% (0.2)
    # In algoritmii genetici reali, pm este de obicei mai mic (ex: 0.5)
    o = mutatie_populatie(p, 18, 12, 0.2)

    grafic.title("Efectul Mutatiei asupra Calitatii Populatiei")
    grafic.xlabel("Index Individ")
    grafic.ylabel("Calitate (Fitness)")
    grafic.legend()
    grafic.show()
