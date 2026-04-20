import copy
import math

import numpy as np

def baza10(x):
    nr_el = len(x)
    y = ''
    for i in range(nr_el):
        y += str(x[i])
    n = int(y,  2)
    return n


def binar(x, nr_el):
    rep = bin(x)[2:]
    rep_f = rep.zfill(nr_el)
    # x_tr = [int(rep_f[i]) for i in range(nr_el)]
    x_tr = []
    for i in range(nr_el + 1):
        x_tr.append(int(rep_f[i]))
    return x_tr

    # print(rep)


def gen_pop(dim, ia, ib):
    populatie = []
    for i in range(dim):
        x = np.random.randint(ia[0], ia[1]+1)
        y = np.random.randint(ib[0] + 1, ib[1] + 2)
        xx = binar(x, 11)
        yy = binar(y, 12)
        individ = xx + yy
        n = fitness(individ)
        individ.append(n)
        populatie.append(individ)
    return populatie


def fitness(individ):
        xx_10 = baza10(individ[0:11])
        yy_10 = baza10(individ[11:23])
        rez = (yy_10 - 1)*math.pow(math.sin(xx_10-2), 2)
        return rez


def recombinare_3p(p1, p2):
    n = 23
    i1 = np.random.randint(0, n - 2)
    i2 = np.random.randint(i1 + 1, n - 1)
    i3 = np.random.randint(i2 + 1, n)
    c1 = p1.copy()
    c2 = p2.copy()
    c1[i1:i2] = p2[i1:i2]
    c2[i1:i2] = p1[i1:i2]
    c1[i2:i3] = p2[i3:]
    c2[i2:i3] = p1[i3:]
    return c1, c2

def recombinare(populatie, pc):
    dim = len(populatie)
    pop_r = copy.deepcopy(populatie)
    for i in range(0, dim-1, 2):
        r = np.random.uniform(0, 1)
        if r < pc:
            p1 = populatie[i][:23].copy()
            p2 = populatie[i+1][:23].copy()
            c1, c2 = recombinare_3p(p1, p2)
            n1 = fitness(c1)
            c1.append(n1)
            n2 = fitness(c2)
            c2.append(n2)
            pop_r[i] = c1
            pop_r[i + 1] = c2
            return pop_r


if __name__ == '__main__':
    dim = 23
    ia = [0, 1500]
    ib = [-1, 2500]
    pop = gen_pop(ia, ib)
    pop_r = recombinare(pop, 0.0)

