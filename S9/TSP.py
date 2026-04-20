import copy
import numpy as np
import random
import matplotlib.pyplot as grafic


def functie_fitness(cost,p):
    cost_p = 0
    for i in range(len(p)-1):
        cost_p += cost[p[i],p[i+1]]
    cost_p += cost[p[0],p[len(p)-1]]
    return 100/cost_p

def mutatie_inversiune(p):
    n = p.size
    i = np.random.randint(0,n-1)
    j = np.random.randint(i+1,n)
    r = p.copy()
    r[i:j+1]=[p[k] for k in range(j,i-1,-1)]
    return r

def mutatie_populatie(pop, val_pop, cost, pm):
    pop_m = pop.copy()
    val_pop_m = val_pop.copy()
    dim = pop_m.shape[0]
    n = len(val_pop_m)
    for i in range(n):
        r = np.random.uniform(0,1)
        if r <= pm:
            x = pop_m[i]
            y = mutatie_inversiune(x)
            pop_m[i] = y
            val_pop_m[i] = functie_fitness(cost,y)
    return pop_m, val_pop_m


def crossover_populatie(pop,valori,cost,pc):
    # initializeaza populatia de copii, po, cu matricea cu elementele 0
    dim=pop.shape[0]
    n=pop.shape[1]
    po=np.zeros((dim,n),dtype="int")
    # initializeaza valorile populatiei de copii, val, cu matricea cu elementele 0
    val=np.zeros(dim,dtype="float")
    #populatia este parcursa astfel incat sunt selectati aleator cate 2 indivizi - matricea este accesata dupa o permutare a multimii de linii 0,2,...,dim-1
    #poz=np.random.permutation(dim)
    #sau populatia este parcursa astfel incat sunt selectati 2 indivizi consecutivi
    poz=range(dim) #- pentru pastrarea ordinii
    for i in range(0,dim-1,2):
        #selecteaza parintii
        x = pop[poz[i]]
        y = pop[poz[i+1]]
        r = np.random.uniform(0,1)
        if r<=pc:
            # crossover x cu y - PMX - potrivit pentru probleme cu dependenta de adiacenta
            c1,c2 = crossover_PMX(x,y)
            v1=functie_fitness(cost,c1)
            v2=functie_fitness(cost,c2)
        else:
            # recombinare asexuata
            c1 = x.copy()
            c2 = y.copy()
            v1=valori[poz[i]]
            v2=valori[poz[i+1]]
        #copiaza rezultatul in populatia urmasilor
        po[i] = c1.copy()
        po[i+1] = c2.copy()
        val[i]=v1
        val[i+1]=v2
    return po, val

def crossover_PMX(x,y):
    n=x.size
    p1=np.random.randint(0,n-1)
    p2=np.random.randint(p1+1,n)
    copil1=PMX(x,y,p1,p2)
    copil2=PMX(y,x,p1,p2)
    return copil1, copil2

def PMX(x,y,p1,p2):
    n=x.size
    copil=-np.ones(n,dtype="int")
    copil[p1:p2+1]=x[p1:p2+1]
    #pozitiile alelelor din secventa de crossover din y care nu apartin lui copil
    poz_alele_neplasate=[k for k in range(p1,p2+1) if not y[k] in copil]
    for p in poz_alele_neplasate:
        a=y[p]
        while copil[p]>-1:
            p=np.where(y==copil[p])[0][0]
        copil[p]=a
    alele_neplasate=[y[k] for k in range(n) if not y[k] in copil]
    pozitii_libere=[k for k in range(n) if copil[k]==-1]
    copil[pozitii_libere]=alele_neplasate.copy()
    return copil


def genereaza_populatie(fc, dim):
    cost = np.genfromtxt(fc)
    n= len(cost)
    pop = np.zeros([dim,n], dtype=int)
    val_pop = np.zeros(dim,dtype=float)
    for i in range(dim):
        pop[i]=np.random.permutation(n)
        val_pop[i]=functie_fitness(cost,pop[i,:n])
    return pop, val_pop, cost


#selectia SUS cu distributia fps cu sigma scalare
#I: pop,qual,dim,n - matricea populatiei, vectorul calitatilor, pop-dim x n, qual - dim
#E: spop,squal - populatia selectata, impreuna cu calitatile membrilor sai
def SUS(pop,qual,dim,n):
    spop=pop.copy()
    squal=np.zeros(dim)
    #utilizeaza fps cu sigma-scalare
    qfps=sigmafps(qual,dim)
    r=np.random.uniform(0,1/dim)
    k,i=0,0
    while (k<dim):
        while (r<=qfps[i]):
            spop[k][:]=pop[i][:]
            squal[k]=qual[i]
            r=r+1/dim
            k=k+1
        i=i+1
    return spop, squal

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
    pm = 0.2
    pc = 0.8
    dim = 30
    p, v, cost = genereaza_populatie("costuri_TSP.txt",dim)
    print(f"Cel mai bun individ initial : {p[np.argmax(v)]}, valoare {np.max(v)}")
    pop_m, val_pop_m = mutatie_populatie(p, v, cost, pm)
    print(f"Cel mai bun individ pop mutatie : {pop_m[np.argmax(val_pop_m)]}, valoare {np.max(val_pop_m)}")
    pop_c, val_pop_c = crossover_populatie(p, v, cost, pc)
    print(f"Cel mai bun individ pop recombinare : {pop_c[np.argmax(val_pop_c)]}, valoare {np.max(val_pop_c)}")

    parinti, valori = SUS(p, v, dim, len(cost))
    genu, valori = elitism(p, v, pop_m, val_pop_m, dim)