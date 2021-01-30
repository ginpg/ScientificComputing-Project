#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Giselt Parra, 26.609.640
# Diego Sanchez, 26.334.929

import matplotlib.pyplot as plt
import numpy as np
import random

h1 = random.uniform(3, 10)
h2 = random.uniform(3, 10)
p1 = random.uniform(1000, 4000)
p2 = random.uniform(1000, 4000)
s = random.uniform(0, 30)

x = np.linspace(0, s)
cad = "Función de iluminación \n\np1 =" +str(p1)+', h1 = '+str(h1) +' \np2 = '+str(p2) + ', h2 = '+str(h2)+'\n'
limiteError = 1e-10

def funL1(x, p, h):
    res = []
    for xval in x:
        res.append([xval,((p * h) / np.sqrt((h**2 + xval**2)**3))])
    return res

def funL2(x, p, h):
    res = []
    for xval in x:
        res.append([xval,((p * h) / np.sqrt((h**2 + (s - xval)**2)**3))])
    return res

#AH YA YO CREO QUE SE QUEDA PEGADA CUANDO LA CONDICION NO SE CUMPLE, NO HAY INTERSECCION
#REVISAR
def inter(x):
    return '((p1 * h1) / np.sqrt((h1**2 + x**2)**3))-((p2 * h2) / np.sqrt((h2**2 + (s - x)**2)**3))'

def evaluar(f,x):
    return eval(f)

#   CERO DE FUNCION: METODO DE BISECCION
def zerofun(f,a,b):
    i=0
    while abs(a-b)>limiteError:
        i+=1
        c = (a+b)/2 
        if (i==1):
            c1 = c
        #plt.plot(c,0,'*',color = 'R')
        fc = evaluar(f,c)
        if fc == 0.0:
            break
        elif evaluar(f,b)*fc < 0.0:
            a = c
        elif evaluar(f,a)*fc < 0.0:
            b = c
        else:
            c = zerofun(f,0,c)
            if c == -1:
                c = zerofun(f,c,b)
            if c == -1:
                return -1
            fc = evaluar(f,c)
        if abs(fc) < limiteError:
            break
    return c1,c

# C'(x)
def dif(x):
    #return '-3*p1*h1*x/(x**2+h1**2)**(5/2) + 3*p2*h2*(s-x)/((s-x)**2+h2**2)**(5/2)'
    #return '((3*p1*h1*x*(x**2+h1**2)**2)/((h1**2+x**2)**3)**(3/2)) + ( (3*h2*p2*((-x+s)**2 + h2**2)**2 * (-x+s)) / ((h2**2 + (s-x)**2)**3)**(3/2))'
    return '((3*h2*p2*(s-x))/((s-x)**2+h2**2)**5/2) - ((3*h1*p1*x))/(x**2+h1**2)**(5/2)'
# C(x)
def funC(x):
    return '((p1 * h1) / np.sqrt((h1**2 + x**2)**3)) + ((p2 * h2) / np.sqrt((h2**2 + (s - x)**2)**3))'

def sortSecond(val): 
    return val[1]



#   CALCULO DE VALORES APROXIMADOS EN LAS FUNCIONES
L1 = np.array(funL1(x, p1, h1))
L2 = np.array(funL2(x, p2, h2))
C = np.array(np.sum([L1, L2], axis=0))

#print(C)

yC = [sub[1] for sub in C]
yL1 = [sub[1] for sub in L1]
yL2 = [sub[1] for sub in L2]
    

    
#   PUNTOS
xmin = C[yC.index(min(yC))][0]/2
xmax = C[yC.index(max(yC))][0]/2

print("xmin aproximado: ",xmin) 
print("xmax aproximado: ",xmax)

if max(yL1)>=min(yL2) and max(yL2)>=min(yL1):
    xeq = zerofun(inter(x),0,s)[1]
    print("xeq aproximado: ",xeq)
    print("\nxmin - xeq:",abs(xmin-xeq))
else:
    print("No existe xeq")


plt.title(cad)
plt.plot(x, yC, label='$C(x)$')
plt.plot(x, yL1, label='$L_1(x)$')
plt.plot(x, yL2, label='$L_2(x)$')

if max(yL1)>=min(yL2) and max(yL2)>=min(yL1):
    plt.plot(xeq,0,'*', label='$xeq$')
plt.plot(x, eval(dif(x)), label='$Derivada(x)$')
plt.grid(b=True)

# VALORES CRITICOS CALCULADOS 
cri = []
cri.append([0, evaluar(funC(x),0)])
cri.append([s, evaluar(funC(x),s)])

c,critico = zerofun(dif(x),0,s)
#plt.plot(critico,0,'*',color = 'b')
cri.append([critico, evaluar(funC(x),critico)])

if critico < c:
    if evaluar(dif(x),c/2+c)*evaluar(dif(x),c)<0.0:
        c,critico = zerofun(dif(x),c,s)
else:
    if evaluar(dif(x),c-c/2)*evaluar(dif(x),c)<0.0:
        c,critico = zerofun(dif(x),0,c)
#plt.plot(critico,0,'*',color = 'b')
cri.append([critico, evaluar(funC(x),critico)])

if critico < c:
    if evaluar(dif(x),c/2+c)*evaluar(dif(x),c)<0.0:
        c,critico = zerofun(dif(x),c,s)
else:
    if evaluar(dif(x),c-c/2)*evaluar(dif(x),c)<0.0:
        c,critico = zerofun(dif(x),0,c)
#plt.plot(critico,0,'*',color = 'b')
cri.append([critico, evaluar(funC(x),critico)])

#print(np.matrix(cri))
cri.sort(key=sortSecond)


print("\nxmin calculado:", cri[0][0])
print("xmax calculado:", cri[4][0])
plt.plot(cri[0][0],0,'*',label='$xmin$')
plt.plot(cri[4][0],0,'*',label='$xmax$')
    
if max(yL1)>=min(yL2) and max(yL2)>=min(yL1):
    xeq = zerofun(inter(x),0,s)[1]
    print("xeq calculado:", xeq)
    print("\nxmin - xeq:",abs(xmin-xeq))



plt.xlabel('Carretera (int)')
plt.ylabel('Iluminación (W)')

plt.legend()
plt.show()
    