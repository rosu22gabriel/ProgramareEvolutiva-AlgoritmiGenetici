import numpy as np
import matplotlib.pyplot as grafic
import random

def calculeaza_functie_obiectiv(n,x):
    c = n * (n-1)/2
    for i in range(n-1):
        for j in range(i+1,n):
            if(abs(i-j)==abs(x[i]-x[j])):
                c-=1
    return c

def mutatie_inserare(p,n):
    p1 = np.random.randint(0,n-1)
    p2 = np.random.randint(p1+1,n)
    rez = p.copy()
    if p1+1<p2:
        el = p[p2]
        rez = np.delete(rez,p2)
        rez = np.insert(rez,p1+1,el)
    return rez


def mutatie_populatie(pop,dim,n,pm):
    pop_m = pop.copy()
    for i in range(dim):
        r = np.random.uniform(0,1)
        if r<=pm:
            x = mutatie_inserare(pop_m[i,:n],n)
            pop_m[i,:n] = x.copy()
            pop_m[i,n] = calculeaza_functie_obiectiv(n,x)
    return pop_m

def crossover_CX(x,y,n):
    ciclu=cicluri(x,y,n)
    c1=x.copy()
    c2=y.copy()
    for i in range(n):
        cat, rest = np.divmod(ciclu[i], 2)
        #sunt interschimbate alelele din ciclurile pare
        #primul ciclu este etichetat cu 1
        if not rest:
            c1[i]=y[i]
            c2[i]=x[i]
    return c1,c2

#calculul ciclurilor din
#I: x,y de dimensiune n
#in
#E: vectorul ciclu, unde ciclu[i]=nr. ciclului din care fac parte x[i] si y[i]
def cicluri(x,y,n):
    #numarul ciclului
    index=1
    ciclu=np.zeros(n)
    gata=0
    while not gata:
        p=np.where(ciclu==0)
        #daca exista gena inca neasignata unui ciclu
        if np.size(p):
            i=p[0][0]
            a=x[i]
            ciclu[i]=index
            b=y[i]
            while b!=a:
                r=np.where(x==b)
                j=r[0][0]
                ciclu[j]=index
                b=y[j]
            index+=1
        else:
            gata=1
    return ciclu

def crossover_populatie(pop,pc):
    #initializeaza populatia de copii, po, cu populatia parintilor
    dim=pop.shape[0]
    m=pop.shape[1]
    n=m-1
    po=pop.copy()
    #populatia este parcursa astfel incat sunt selectati indivizii 0,1 apoi 2,3 s.a.m.d
    for i in range(0,dim-1,2):
        #selecteaza parintii
        x = pop[i,:-1]
        y = pop[i+1,:-1]
        r = np.random.uniform(0,1)
        if r<=pc:
            # crossover x cu y - OCX sau CX - potrivit pentru NRegine
            c1,c2 = crossover_CX(x,y,n)
            val1=calculeaza_functie_obiectiv(n,c1)
            val2= calculeaza_functie_obiectiv(n,c2)
            po[i][:n]=c1.copy()
            po[i][n]=val1
            po[i+1][:n]=c2.copy()
            po[i+1][n]=val2
    return po

def genereaza_populatie(n, dim):
    pop = np.zeros([dim,n+1], dtype=int)
    for i in range(dim):
        pop[i,:n] = np.random.permutation(n)
        pop[i,n] = calculeaza_functie_obiectiv(n,pop[i])
    return pop


#SELECTIA TURNEU pentru k indivizi
#selectia turneu - selecteaza dim indivizi
#I: pop,qual,dim,n,k - matricea populatiei, vectorul calitatilor, pop-dim x n, qual - dim, numarul indivizilor care intra in competitie
#E: spop,squal - populatia selectata, impreuna cu calitatile membrilor sai


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

#selectia ruleta cu distributia fps cu sigma-scalare
#I: pop,qual,dim,n - matricea populatiei, vectorul calitatilor, pop-dim x n, qual - dim
#E: spop,squal - populatia selectata, impreuna cu calitatile membrilor sai
def ruleta(pop,qual,dim,n):
    spop=pop.copy()
    squal=np.zeros(dim)
    #utilizeaza fps cu sigma-scalare
    qfps=sigmafps(qual,dim)
    for it in range(dim):
        #selectia unui individ la iteratia it
        r=np.random.uniform(0,1)
        poz=np.where(qfps>=r)
        #isel = prima pozitie in qfps cu suma primelor isel elemente mai mare decat r
        isel=poz[0][0]
        spop[it][:]=pop[isel][:]
        squal[it]=qual[isel]
    return spop, squal

#fps cu sigma scalare
# I: qual,dim - vectorul calitatilor, de dimensiune dim
# E: distributia cumulata qfps
#daca toate elementele din qual sunt egale, returneaza fps standard
def sigmafps(qual, dim):
    #media vectorului qual
    med=np.mean(qual)
    #deviatia standard - qual
    var=np.std(qual)
    #calculul variantei sigma-scalate: newq[i]=max(qual[i]-(med-2var)), i=1...dim
    newq=[max(0,qual[i]-(med-2*var)) for i in range(dim)]
    #calculul distributieipe noul vector
    if np.sum(newq)==0:
        qfps=fps(qual,dim)
    else:
        qfps=fps(newq,dim)
    return qfps

#calculul distributiei de probabilitate FPS pentru un vector de calitati
# I: qual,dim - vectorul calitatilor, de dimensiune dim
# E: distributia cumulata qfps
def fps(qual,dim):
    fps=np.zeros(dim)
    #suma valorilor vectorului qual
    suma=np.sum(qual)
    for i in range (dim):
        fps[i] = qual[i]/suma
    qfps=fps.copy()
    for i in range(1, dim):
        qfps[i]=qfps[i-1]+fps[i]
    return qfps

if __name__ == "__main__":
    n = 8
    dim = 30
    pm = 0.25
    pc = 0.8
    p = genereaza_populatie(8,30)
    print(f"Cel mai bun individ initial : {p[np.argmax(p[:,-1]),:n]}, valoare {np.max(p[:,-1])}")
    pop_m  = mutatie_populatie(p,dim, n, pm )
    print(f"Cel mai bun individ mutatie : {pop_m[np.argmax(pop_m[:,-1]),:n]}, valoare {np.max(pop_m[:,-1])}")
    pop_c  = crossover_populatie(p, pc )
    print(f"Cel mai bun individ mutatie : {pop_c[np.argmax(pop_c[:,-1]),:n]}, valoare {np.max(pop_c[:,-1])}")
    parinti, valori = ruleta(p[:, :n], p[:, n], dim, n)

    genu, valori = elitism(p[:, :n], p[:, n], pop_m[:, :n], pop_m[:, n], dim)