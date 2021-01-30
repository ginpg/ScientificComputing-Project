#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Giselt Parra, 26.609.640
# Diego Sanchez, 26.334.929

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import numpy as np
import scipy as sp
import random


h1 = random.uniform(3, 10) 
p1 = random.uniform(1000, 4000)
p2 = random.uniform(1000, 4000)
s = random.uniform(0, 30)
cad = "Función de iluminación \n\np1 =" +str(p1)+', h1 = '+str(h1) +' \np2 = '+str(p2)
limiteError = 1e-10


def funC(x,h):
    return (h1*p1)/(x**2+h1**2)**(3/2)+(h*p2)/((s-x)**2+h**2)**(3/2)

def difuno(x,h):
    return (3*h*p2*(s-x))/((s-x)**2+h**2)**(5/2)-(3*h1*p1*x)/(x**2+h1**2)**(5/2)

#   CERO DE FUNCION: METODO DE BISECCION
def zerofun(h2,a,b):
    i = 0
    while abs(a-b)>limiteError:
        i+=1
        c = (a+b)/2 
        if (i==1):
            c1 = c
        fc = difuno(c,h2)
        if fc == 0.0:
            break
        elif difuno(b,h2)*fc < 0.0:
            a = c
        elif difuno(a,h2)*fc < 0.0:
            b = c
        else:
            c = zerofun(h2,0,c)
            if c == -1:
                c = zerofun(h2,c,b)
            if c == -1:
                return -1
            fc = difuno(c,h2)
        if abs(fc) < limiteError:
            break
    return c1, c



def sortFirst(val): 
    return val[0]

def sortSecond(val): 
    return val[1]

def sortThird(val): 
    return val[2]

def graficar(li, ls, do):
    
    x = np.linspace(li, ls)
    h2 = np.linspace(3, 10)
    
    #   CALCULO DE VALORES APROXIMADOS EN LAS FUNCIONES
    minimos = []
    maximos = []
    for i in h2:
        # VALORES CRITICOS CALCULADOS 
        cri = []
        cri.append([0, funC(0,i),i])
        cri.append([s, funC(s,i),i])
        
        c,critico = zerofun(i,0,s)
        cri.append([critico,funC(critico,i),i])
        
        if critico < c:
            if (difuno(c/2+c,i)*difuno(c,i))<0.0:
                c,critico = zerofun(i,c,s)
        else:
            if (difuno(c-c/2,i)*difuno(c,i))<0.0:
                c,critico = zerofun(i,0,c)
        cri.append([critico,funC(critico,i),i])
        
        if critico < c:
            if (difuno(c/2+c,i)*difuno(c,i))<0.0:
                c,critico = zerofun(i,c,s)
        else:
            if (difuno(c-c/2,i)*difuno(c,i))<0.0:
                c,critico = zerofun(i,0,c)
        cri.append([critico,funC(critico,i),i])
        
        cri.sort(key=sortSecond)
        minimos.append(cri[0])
        maximos.append(cri[4])
    
    a = minimos
    minimos = np.array(minimos)
    xminimos = [sub[0] for sub in minimos]
    yminimos = [sub[1] for sub in minimos]
    ymaximos = [sub[1] for sub in maximos]
    ymaximos.sort()
   
    a.sort(key=sortSecond)
    minmax = a[len(a)-1] #EL MAYOR DE LOS MENORES
    
    h21d = h2
    x, h2 = np.meshgrid(x, h2)
    C = np.array(funC(x, h2))
    #D = np.array(difuno(x, h2))
    
    # Grafico 3d
    fig = plt.figure(figsize = (7,5))
    ax = fig.gca(projection='3d')
    ax.plot_surface(x, h2, C,rstride=1, cstride=1, cmap=plt.cm.coolwarm)
    #ax.plot_surface(x, h2, D,rstride=2, cstride=2)
    ax.scatter(xminimos, h21d, yminimos, c='m',label = 'puntos mínimos')
    ax.scatter(minmax[0], minmax[2], minmax[1], c='black',label = '$(x^0, h_2^0)$')
    ax.contour(x, h2, C, zdir='z', offset=0,cmap=plt.cm.coolwarm)
    ax.set_xlabel('Carretera(int)')
    ax.set_ylabel('h2')
    ax.set_zlabel('Iluminacion(W)')
    ax.set_xlim(li,ls)
    ax.set_ylim(3,10)
    a.sort(key=sortSecond)
    if do:
        m = (a[len(a)-1][1] - a[0][1])/2
        ax.set_zlim(a[0][1]-m,a[len(a)-1][1]+m)
    else: 
        ax.set_zlim(0,ymaximos[len(ymaximos)-1])
    fig.tight_layout()
    plt.title(cad)
    plt.legend()
    plt.show()

    if do:
        
        # h2 x X
        fig = plt.figure(figsize = (7,5))
        a.sort(key=sortThird)
        plt.title(cad)
        plt.plot([sub[2] for sub in a], [sub[0] for sub in a],'--',color='m')
        plt.xlabel('h2')
        plt.ylabel('Carretera(int)') 
        plt.grid(b=True, linestyle="--")
        plt.show()
        
        """
        print("x: ",minmax[0])
        print("W: ",minmax[1])
        print("h2: ",minmax[2])
        """
        print("\nMinimo con mejor iluminacion: \n("+str(minmax[0])+", "+str(minmax[2])+")")
        
    a.sort(key=sortFirst)
    #print(np.matrix(a))
    return a[0][0], a[len(a)-1][0]


a, b= graficar(0,s,0)
if abs(a - b)<=0:
    if a != 0:
        a = a-0.5
    if b != s:
        b = b+0.5
graficar(a,b,1)


