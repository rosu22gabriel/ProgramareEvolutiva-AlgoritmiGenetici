import numpy as np

def functie_exemplu():
    """Acesta este un docstring!"""
    print("Aceasta este o procedura")

# def functie_exemplu_eroare_indentare():
# """Acesta este un docstring!"""
# print("Acesta este o procedura")


if __name__ == '__main__':
    # print("Test")
    #
    # # Exemplu Afisare
    # print("Acesta este un script Python")
    #
    # # Exemplu Afisare Docstring
    # print(functie_exemplu().__doc__)
    # functie_exemplu()
    #
    # x = 10
    # y = 3.14
    # nume = "Alex"
    # print(x, y, nume)
    #
    # a = 5
    # b = float(a)
    # c = str(a)
    # print (a, b, c)
    #
    # # Lista
    # fructe = ["mere", "pere", "banane"]
    # fructe.append("struguri")
    # print(fructe[1])
    # print(fructe[-1])
    #
    # # Tuplu
    # zile = ("Luni", "Marti", "Miercuri", "Marti")
    # print(zile[-2])
    #
    # # Dictionar
    # student = {"nume": "Alex", "varsta": 20, "nota": 9.5}
    # print(student["nume"])
    # print(len(student))
    #
    # # Operatori aritmetici
    # a, b = 10, 3
    # print(a + b)
    # print(a % b)
    # print(a ** b)
    #
    # # Operatori de atribuire
    # x = 5
    # x += 3
    # print(x)
    #
    # # Operatori de comparatie
    # print( 5 > 3 )
    # print( 10 == 5)
    #
    # # Operatori pe biti
    # a, b = 5, 3
    # print( a & b )
    # print( a | b)
    #
    # nume = input("Introdu numele:")
    # print(type(nume))
    # varsta = int(input("Introdu varsta: "))
    #
    # print("Salut", "Python", sep=" --- ")
    # print("Final de linie", end=" *** ")

    # with open("fisier.txt", "w") as f:
    #     f.write("Hello, world!")
    #
    # with open("fisier.txt", "r") as f:
    #     continut = f.read()
    #     print(continut)
    #
    # num, *num2 = [7, 2, 3, 4, 5]
    # if num % 2 == 0:
    #     print(f"{num} este par")
    # else:
    #     print(f"{num} este impar")
    #
    # print("Numere pare intre 1 si 10:")
    # for i in range(1, 11):
    #     if i % 2 == 0:
    #         print(i)

    # x = 5
    # print("Countdown:")
    # while x > 0:
    #     print(x, end=" ") # Output: 5 4 3 2 1
    #     x -= 1
    # print("\nFinalizat!")

    # aa = np.array(range(1, 10, 2))
    # print(aa)

    array = np.array([1, 2, 3, 4, 5])
    matrix = np.array([[1, 2, 3], [4, 5, 6]])

    # print(array)
    # print(matrix)
    #
    # print(matrix.shape)
    # print(matrix.dtype)
    # print(matrix.size)
    # print(matrix.ndim)
    #
    # zeros = np.zeros((2, 3))
    # ones = np.ones((3, 3))
    # random_arr = np.random.rand(2, 2)
    # range_arr = np.arange(0, 10, 2)
    #
    # print(zeros)
    # print(ones)
    # print(random_arr)
    # print(range_arr)
    #
    # print(matrix[0, -1])
    # print(matrix[:, 1])
    # print(matrix[1, :])

    # arr = np.arange(10)
    # print(arr[::2])
    #
    # arr = np.array([10, 20, 30, 40, 50])
    # index = [0, 2, 4]
    # print(arr[index])
    #
    # arr = np.array([10, 20, 30, 40, 50])
    # print(arr[arr > 25])
    #
    # A = np.array([[1, 2], [3, 4]])
    # B = np.array([[5, 6], [7, 8]])
    # produs = np.dot(A, B)
    # print(produs)

    # arr = np.arange(6).reshape(2, 3)
    # print(arr)
    # print(arr.T)

    # creare vector de doua elemente
    # v = np.random.choice([2, 3, 9, 11, 100, 22], 2)
    # print(v)

    # generare vector 3 elemente intre -3, 11
    # v = np.random.randint(-3, 11, 3)
    # suma = np.sum(v[v > 1])
    # print(v)
    # print("Suma ", suma)

    ## generare matrice[3, 5] cu elemente distribuite uniform intre -2, 7
    # a = np.random.uniform(-2, 7, [3, 5])
    # print(a)
    # maxim_a = a.max()
    # print("Maxim ", maxim_a)

    ## citire din fisier
    v = np.genfromtxt("matrice.txt", int)
    print("Linii si coloane ", v.shape)

    # sortare matrice
    v_sort = np.sort(v, axis=None).reshape(v.shape)
    print("Sortare matrice", v_sort, sep="\n")

    # valoare medie
    val_medie = np.mean(v)
    print("Valoare medie", val_medie)

    # deviatie standard
    dev_std = np.std(v)
    print("Deviatie standard", dev_std)

    # cautare valoarea 5 in matrice
    poz1 = np.argwhere(v_sort == 5)
    print("Pozitie valoare 5 ", poz1.flatten())

    # sortare matrice dupa ultima coloana
    indici = v_sort[:, -1].argsort()
    b = v_sort[indici]
    print("Matrice sortata dupa ultima coloana :",b)

    # copierea unei matrice in fisier text - similar vectorilor
    np.savetxt('matrice_i.txt', v_sort, "%i")
    np.savetxt('matrice_f.txt', v_sort, '%.2f')

