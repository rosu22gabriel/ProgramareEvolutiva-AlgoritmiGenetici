import numpy as np
import random
import matplotlib.pyplot as grafic
import copy

def calculeaza_fitness(cost,valoare,x,Cmax):
    cost_x = np.dot(x,cost)
    valoare_x = np.dot(x,valoare)
    return cost_x<=Cmax, valoare_x


def mutatie_pop (pop, cost, valoare, Cmax, pm):
    pop_m = copy.deepcopy(pop)
    dim = len(pop)
    n = cost.size
    pop_mut = []
    for i in range(dim):
        x = pop_m[i][:n].copy()
        mutat = False
        for j in range(n):
            r = random.uniform(0,1)
            if r <= pm:
                x[j] = int(not x[j])
                mutat = True
        if mutat:
            gasit, val = calculeaza_fitness(cost, valoare, x, Cmax)
            if gasit:
                x = x + [val]
                pop_m[i] = x.copy()
                pop_mut.append(x)
            else:
                pop_mut.append(pop_m[i])
        else:
            pop_mut.append(pop_m[i])
    return pop_m

def crossover_uniform(x,y,n):
    #initializeaza copiii cu valorile parintilor
    c1=x.copy()
    c2=y.copy()
    #constructia copiilor
    for i in range(n):
        r = np.random.randint(0,2)
        # daca r==1 interschimba alelele din i
        if r == 1:
            c1[i] = y[i]
            c2[i] = x[i]
    return c1,c2

def crossover_populatie(pop,cost,vvaloare,Cmax,pc):
    dim=len(pop)
    n=cost.size
    po=copy.deepcopy(pop)
    #populatia este parcursa astfel incat sunt selectati indivizii 0,1 apoi 2,3 s.a.m.d
    for i in range(0,dim-1,2):
        #selecteaza parintii
        x = pop[i][:-1].copy()
        y = pop[i+1][:-1].copy()
        r = np.random.uniform(0,1)
        if r<=pc:
            # crossover x cu y - uniform: mai potrivit aici
            c1,c2=crossover_uniform(x ,y ,n)
            fez, val = calculeaza_fitness(cost,valoare,c1,Cmax)
            if fez:
                po[i][:-1]=c1.copy()
                po[i][-1]=val
            fez, val = calculeaza_fitness(cost,valoare,c2,Cmax)
            if fez:
               po[i+1][:-1]=c2.copy()
               po[i+1][-1]=val
    return po

def genereaza_populatie(cost, valoare, Cmax, dim):
    n = valoare.size
    pop = []
    for i in range(dim):
        gasit = False
        x = []
        while gasit == False:
            x = random.choices([0,1],k = n)
            gasit, val = calculeaza_fitness(cost,valoare,x,Cmax)
        x.append(val)
        pop.append(x)
    return pop


#SELECTIA TURNEU pentru k indivizi
#selectia turneu - selecteaza dim indivizi
#I: pop,qual,dim,n,k - matricea populatiei, vectorul calitatilor, pop-dim x n, qual - dim, numarul indivizilor care intra in competitie
#E: spop,squal - populatia selectata, impreuna cu calitatile membrilor sai
def turneu(pop,qual,dim,n,k):
    spop=pop.copy()
    squal=np.zeros(dim)
    for it in range(dim):
        #selectia unui individ la iteratia it
        #generarea a k indici aleatori in poz
        poz=np.random.randint(0,dim,k)
        #calitatile care intra in competitie
        v=[qual[poz[i]] for i in range(k)]
        M=max(v)
        p=np.where(v==M)
        #imax = prima pozitie in v in care este atinsa valoarea maxima
        imax=p[0][0]
        # isel=corespondentul lui imax in populatie
        isel=poz[imax]
        spop[it][:]=pop[isel][:]
        squal[it]=qual[isel]
    return spop, squal

#selectia elitista
#I: pop_c,qual_c,pop_mo,qual_mo - populatiile curenta si a copiilor mutati pe baza carora se face selectia, fiecare insotita de vectorii calitatilor
#E: pop,qual - populatia rezultata si vectorul calitatilor
def elitism(pop_c,qual_c,pop_mo,qual_mo,dim):
    pop=np.copy(pop_mo)
    qual=np.copy(qual_mo)
    max_c=np.max(qual_c)
    max_mo=np.max(qual_mo)
    if max_c>max_mo:
        p1=np.where(qual_c==max_c)
        #imax = prima pozitie in qual_c in care este atinsa valoarea maxima
        imax=p1[0][0]
        ir=np.random.randint(dim)
        # inlocuirea
        pop[ir]=pop_c[imax].copy()
        qual[ir]=max_c
    return pop,qual

if __name__ == "__main__":
    Cmax = 50
    dim = 10
    pm = 0.2
    pc = 0.8
    k = 2
    cost = np.genfromtxt("cost_disc.txt")
    valoare = np.genfromtxt("valoare_disc.txt")
    pop = genereaza_populatie(cost,valoare,Cmax,dim)
    print(f"Cel mai bun Individ Pop Initiala: {max(pop, key=lambda x: x[-1])}")
    pop_mut =  mutatie_pop(pop, cost, valoare, Cmax, pm)
    print(f"Cel mai bun Individ Pop Mutatie: {max(pop_mut, key=lambda x: x[-1])}")
    pop_recombinare =  crossover_populatie(pop, cost, valoare, Cmax, pc)
    print(f"Cel mai bun Individ Pop Recombinare: {max(pop_recombinare, key=lambda x: x[-1])}")

    pop_1=[pop[i][:-1] for i in range(len(pop))]
    cal=[pop[i][-1] for i in range(len(pop))]
    parinti,valori=turneu(pop_1,cal,dim,valoare.size,2)

    pop_mut_1 = [pop_mut[i][:-1] for i in range(len(pop_mut))]
    cal_mut = [pop_mut[i][-1] for i in range(len(pop_mut))]
    genu,valori=elitism(pop_1,cal,pop_mut_1,cal_mut,dim)
