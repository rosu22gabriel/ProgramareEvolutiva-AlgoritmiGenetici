import numpy as np
import matplotlib.pyplot as grafic
import random
import copy

def calculeaza_fitness(cost, valoare, Cmax, x):
    cost_x = sum([x[i]*cost[i] for i in range(len(cost))])
    valoare_x = sum([x[i]*valoare[i] for i in range(len(valoare))])
    return cost_x<=Cmax, valoare_x
def mutatie_populatie(pop,cost, valoare, Cmax, pm, sigma):
    pop_m = copy.deepcopy(pop)
    dim = len(pop)
    n = cost.size
    pop_mut = []
    for i in range(dim):
        x = pop[i][:n].copy()
        mutat = False
        for j in range(n):
            r = random.uniform(0,1)
            if r <= pm:
                p = np.random.normal(0, sigma)
                y = x[j] + p
                if y > 1:
                    y = 1
                if y < 0:
                    y = 0
                x[j]=y
                mutat = True
        if mutat:
            fez, val = calculeaza_fitness(cost, valoare, Cmax, x)
            if fez:
                x = x + [val]
                pop_m[i] = x.copy()
                pop_mut.append(x)
            else:
                pop_mut.append(pop_m[i])
        else:
            pop_mut.append(pop_m[i])
    return pop_m


def genereaza_populatie(cost,valaore,Cmax,dim):
    n = np.size(cost)
    pop = []
    for i in range(dim):
        gasit = False
        while  not gasit:
            x = [random.uniform(0,1) for _ in range(n)]
            gasit, val_x = calculeaza_fitness(cost, valoare, Cmax, x)
        x=x+[val_x]
        pop = pop + [x]
    return pop

def crossover_singular(x ,y ,n,alpha):
    c1=x.copy()
    c2=y.copy()
    i=random.randint(0,n-1)
    c1[i]=alpha*x[i]+(1-alpha)*y[i]
    c2[i]=alpha*y[i]+(1-alpha)*x[i]
    return c1,c2


def crossover_populatie(pop,cost,valoare,Cmax,pc, alpha):
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
            c1,c2=crossover_singular(x ,y ,n,alpha)
            fez, val = calculeaza_fitness(cost,valoare,Cmax, c1)
            if fez:
                po[i][:-1]=c1.copy()
                po[i][-1]=val
            fez, val = calculeaza_fitness(cost,valoare,Cmax, c2)
            if fez:
               po[i+1][:-1]=c2.copy()
               po[i+1][-1]=val
    return po

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
    alpha = 0.3
    cost = np.genfromtxt("cost_cont.txt")
    valoare = np.genfromtxt("valoare_cont.txt")

    pop = genereaza_populatie(cost,valoare , 50, 10)
    print(f"Cel mai bun Individ Pop Initiala: {max(pop, key=lambda x: x[-1])}")
    pop_mut  = mutatie_populatie(pop, cost, valoare, Cmax, 0.8,0.7)
    print(f"Cel mai bun Individ Pop Mutatie: {max(pop_mut, key=lambda x: x[-1])}")
    pop_recombinare =  crossover_populatie(pop, cost, valoare, Cmax, pc, alpha)
    print(f"Cel mai bun Individ Pop Recombinare: {max(pop_recombinare, key=lambda x: x[-1])}")

    pop_1=[pop[i][:-1] for i in range(len(pop))]
    cal=[pop[i][-1] for i in range(len(pop))]
    parinti,valori=turneu(pop_1,cal,dim,valoare.size,2)

    pop_mut_1 = [pop_mut[i][:-1] for i in range(len(pop_mut))]
    cal_mut = [pop_mut[i][-1] for i in range(len(pop_mut))]
    genu,valori=elitism(pop_1,cal,pop_mut_1,cal_mut,dim)
