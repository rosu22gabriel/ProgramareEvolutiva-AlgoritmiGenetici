# creare lista
# a. lista vida
L = []
print(L)

# b. lista cu N elemente, fara valori
N = 10
L = [None]*N
# L = [0]*N - N valori, toate 0
# accesul elementelor
L[0]=L[2]=7
print(L)
# eliminarea tuturor elementelor listei
L.clear()
print(L)

# adaugare de elemente in L
L.append(7)
L = L + [10]
L.append(8)
print(L)

# Stergerea unui element dintr-o pozitie
L.pop(0)
print(L)

# Adaugarea unui element intr-o pozitie
L.insert(2, 7)
print(L)

# Liste comprehensive - exemple

# L = {x / xin M si P(X)}
# PA = [x for x in range(10, 31, 5)]

# generarea unei liste care contine o progresie aritmetica 45...-34 cu ratia -5

L=[i for i in range(45, -34, -5)]

# selectarea elementelor divizibile cu k din lista
k = 3
LP = [x for x in L if x % 3 == 0]
# calculul maximului din lista si determinarea pozitiei de aparitie
maxim = max(L)
index = L.index(maxim)
print("Lista L:", L)
print("Elementele divizibile cu 3 din L:", LP)
print("\nMaximul este ", maxim, " si apare in pozitia ", index)

# toate pozitiile de aparitie ale unui element intr-o lista
t = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 3, 2, 1]
toate_poz=[i for i, j in enumerate(t) if j == 3]
# i pozitie in t, j valoare
print("\n3 apare in lista", t)
print("in pozitiile:", toate_poz)

# generarea unei multimi de tipul -1, -1+eps, -1+2*eps, ..., 1
eps = 0.1
L1=[-1+i*eps for i in range(10000) if -1+i*eps <= 1]
print("\n Valorile din [-1, 1] cu pas 0.1", L1)

# alternativ, folosind numpy
import numpy
L2 = numpy.arange(-1, 1 + 0.1, 0.1)
print("\nValorile din [-1, 1] cu pas 0.1", L2)

# calculul produsului scalar dintre 2 liste generate aleator
# listele au cate 10 elemente, fiecare element in {-2, 3, 7, 2.5}
import random
l1=[random.choice([-2,3,7,2.5]) for _ in range(10)]
l2=[random.choice([-2,3,7,2.5]) for _ in range(10)]
dot_prod=sum(x*y for x,y in zip(l1, l2))
print(f"\n\nProdusul scalar dintre {l1} si {l2} este {dot_prod:.3f} ")