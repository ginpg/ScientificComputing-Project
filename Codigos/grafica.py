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
limiteError = 1e-8
 
def inter(f,x):
    return eval(f)

def c():
    return ((p1 * h1) / np.sqrt((h1**2 + x**2)**3)), ((p2 * h2) / np.sqrt((h2**2 + (s - x)**2)**3))

def funL1(x, p, h):
    res = []
    for xval in x:
        res.append((p * h) / np.sqrt((h**2 + xval**2)**3))
    return res

def funL2(x, p, h):
    res = []
    for xval in x:
        res.append((p * h) / np.sqrt((h**2 + (s - xval)**2)**3))
    return res

def inter(x):
    return ((p1 * h1) / np.sqrt((h1**2 + x**2)**3))-((p2 * h2) / np.sqrt((h2**2 + (s - x)**2)**3))

#   CERO DE FUNCION: METODO DE BISECCION
def zerofun(f,a,b):
    i=0
    while True:
        i+=1
        c = (a+b)/2 
        fc = inter(c)
        if fc == 0.0:
            break
        elif inter(a)*fc < 0.0:
            b = c
        else:
            a = c
        if abs(fc) < limiteError:
            break
    return c

#   CALCULO DE VALORES EN LAS FUNCIONES
L1 = funL1(x, p1, h1)
L2 = funL2(x, p2, h2)
C = np.sum([L1, L2], axis=0)
f = inter(x)

#   PUNTOS
xmin = min(C) #El valor real es realidad es C(x)= xmin
xmax = max(C) #El valor real en realidad es C(x)= xmax
xeq = zerofun(f,0,s)


#print(C)

print("xmin: ",xmin) 
print("xmax: ",xmax)
print("xeq: ",xeq)


plt.plot(x, L1, label='$L_1(x)$')
plt.plot(x, L2, label='$L_2(x)$')
plt.plot(x, C, label='$C(x)$')
plt.plot(x, f, label='$L_1-L_2$')
plt.xlabel('Carretera (int)')
plt.ylabel('Iluminación (W)')
plt.title(cad)
plt.grid(b=True)
plt.legend()
plt.show()
