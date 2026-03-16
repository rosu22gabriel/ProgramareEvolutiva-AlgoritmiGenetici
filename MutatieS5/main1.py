import random


def calculeaza_calitate_p1(p):
    """
    Calculează numărul de perechi (i, j) unde i < j și p[i] < p[j].
    Aceasta este funcția obiectiv f(p).
    """
    n = len(p)
    f_p = 0
    for i in range(n):
        for j in range(i + 1, n):
            if p[i] < p[j]:
                f_p += 1
    return f_p


def genereaza_populatie_p1(n, dim):
    """
    Generează o populație de dimensiune 'dim', unde fiecare individ
    este o permutare de n elemente.
    """
    populatie = []
    # Elementele de bază ale permutării (de la 1 la n)
    elemente_baza = list(range(1, n + 1))

    for _ in range(dim):
        # Generăm o permutare aleatorie
        indiv = random.sample(elemente_baza, n)
        # Calculăm calitatea (fitness)
        calitate = calculeaza_calitate_p1(indiv)
        # Memoram calitatea la sfârșitul reprezentării
        indiv.append(calitate)
        populatie.append(indiv)

    return populatie


def mutatie_swap(populatie, pm):
    """
    Aplică operatorul de mutație prin interschimbare pe întreaga populație.
    """
    pop_mutație = []
    # n este numărul de elemente (fără valoarea de fitness de la final)
    n = len(populatie[0]) - 1

    for individ in populatie:
        # Lucrăm pe o copie a individului (fără fitness-ul vechi)
        nou_individ = individ[:-1]

        # Verificăm dacă are loc mutația pe baza probabilității pm
        if random.random() < pm:
            # Alegem doi indici diferiți aleatoriu
            idx1, idx2 = random.sample(range(n), 2)
            # Interschimbăm elementele (Swap)
            nou_individ[idx1], nou_individ[idx2] = nou_individ[idx2], nou_individ[idx1]

        # Recalculăm calitatea pentru noul individ
        noua_calitate = calculeaza_calitate_p1(nou_individ)
        nou_individ.append(noua_calitate)
        pop_mutație.append(nou_individ)

    return pop_mutație