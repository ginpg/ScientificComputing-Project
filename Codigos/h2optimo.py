#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Giselt Parra, 26.609.640
# Diego Sanchez, 26.334.929


# C'(x) = (3*h2*p2*(s-x))/((s-x)**2+h2**2)**(5/2)-(3*h1*p1*x)/(x**2+h1**2)**(5/2)
# C'(h2) = -(p2*(2*h2**2-x**2+2*s*x-s**2))/(h2**2+(s-x)**2)**(5/2)
    
# C''(xx) = -(3*h1*p1)/(x**2+h1**2)^(5/2)+(15*h1*p1*x**2)/(x**2+h1**2)**(7/2)+(15*h2*p2*(s-x)**2)/((s-x)**2+h2**2)**(7/2)-(3*h2*p2)/((s-x)**2+h2**2)**(5/2)
# C''(xh2) -(3*p2*(x-s)*(4*h2**2-x**2+2*s*x-s**2))/(h2**2+(s-x)**2)**(7/2)
# C''(h2h2) = (3*p2*h2*(2*h2**2-3*x**2+6*s*x-3*s**2))/(h2**2+(s-x)**2)**(7/2)
# C''(h2x) = (15*h2*p2*(x-s)*(3*x**2-6*s*x+3*s**2-4*h2**2))/((s-x)**2+h2**2)**(9/2)
    

from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np
import scipy as sp
import random

h1 = random.uniform(3, 10) 
p1 = random.uniform(1000, 4000)
p2 = random.uniform(1000, 4000)
s = random.uniform(0, 30)

x = np.linspace(0, s)
h2 = np.linspace(3, 10)

cad = "Función de iluminación \n\np1 =" +str(p1)+', h1 = '+str(h1) +' \np2 = '+str(p2)
limiteError = 1e-10

def funL1(x):
    return (p1 * h1) / np.sqrt((h1**2 + x**2)**3)

def funL2(x, h):
    return (p2 * h) / np.sqrt((h**2 + (s - x)**2)**3)

def funC(x,h):
    return ((p1 * h1) / np.sqrt((h1**2 + x**2)**3)) + ((p2 * h) / np.sqrt((h**2 + (s - x)**2)**3))

def C(x,h):
    return '((p1 * h1) / np.sqrt((h1**2 + x**2)**3)) + ((p2 * h) / np.sqrt((h**2 + (s - x)**2)**3))'

def difuno(x,h2):
    return -3*p1*h1*x/(x**2+h1**2)**(5/2) + 3*p2*h2*(s-x)/((s-x)**2+h2**2)**(5/2)


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


def dif(v):
    x = v.item(0)
    h = v.item(1)
    return np.matrix([-(3*h*p2*(s-x))/((s-x)**2+h**2)**(5/2)-(3*h1*p1*x)/(x**2+h1**2)**(5/2),-(p2*(2*h**2-x**2+2*s*x-s**2))/(h**2+(s-x)**2)**(5/2)]).T

def diff(v):
    x = v.item(0)
    h = v.item(1)
    return np.matrix(
[[-(3*h1*p1)/(x**2+h1**2)**(5/2)+(15*h1*p1*x**2)/(x**2+h1**2)**(7/2)+(15*h*p2*(s-x)**2)/((s-x)**2+h**2)**(7/2)-(3*h*p2)/((s-x)**2+h**2)**(5/2),-(3*p2*(x-s)*(4*h**2-x**2+2*s*x-s**2))/(h**2+(s-x)**2)**(7/2)],[(3*p2*h*(2*h**2-3*x**2+6*s*x-3*s**2))/(h**2+(s-x)**2)**(7/2), (15*h*p2*(x-s)*(3*x**2-6*s*x+3*s**2-4*h**2))/((s-x)**2+h**2)**(9/2)]])  

def BFGS(x,H,tol,maxiter):
    neg = np.matrix([1,1])
    for i in range(0,maxiter):
        print('\n       ',i)
        p = sp.linalg.solve(H,dif(x)*neg)
        print("p:\n",p)
        xn = x + p
        print("xn:\n",xn)
        Cxn = dif(xn)
        print("Cxn:\n",Cxn)
        Cx = dif(x)
        print("Cx:\n",Cx)
        y = Cxn - Cx
        print("y:\n",y)
        H = (H + y*y.T)/(y.T*p) + Cxn*Cx.T/p.T*Cx
        print("H:\n",H)
        x = xn
        print("x:\n",x)
        print("NORMA: ",np.linalg.norm(Cxn))
        if np.linalg.norm(Cxn)<=limiteError:
            break
    return xn
            
    return xn

def sortFirst(val): 
    return val[0]

def sortSecond(val): 
    return val[1]

def sortThird(val): 
    return val[2]

#   CALCULO DE VALORES APROXIMADOS EN LAS FUNCIONES
L1 = np.array(funL1(x))
L2 = np.array(funL2(x, h2))

fig = plt.figure(figsize = (7,5))
ax = fig.gca(projection='3d')


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
ax.scatter(xminimos, h2, yminimos, c='g',label = 'puntos mínimos')
a.sort(key=sortSecond)
minmax = a[len(a)-1]
print("x: ",minmax[0])
print("W: ",minmax[1])
print("h2: ",minmax[2])

#BFGS
#sp.optimize.minimize(C(x,h2),np.matrix([minmax[0],minmax[2]]).T)
#BFGS(np.matrix([minmax[0],minmax[2]]).T,diff(np.matrix([minmax[0],minmax[2]]).T),1e-10,500)

x, h2 = np.meshgrid(x, h2)
C = np.array(funC(x, h2))

# Grafico 3d
ax.plot_surface(x, h2, C,rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.contour(x, h2, C, zdir='z', offset=0, cmap=plt.cm.coolwarm)
ax.set_xlabel('Carretera(int)')
ax.set_ylabel('h2')
ax.set_zlabel('Iluminacion(W)')
ax.set_xlim(0,s)
ax.set_ylim(3,10)
ax.set_zlim(0,ymaximos[len(ymaximos)-1])
fig.tight_layout()
plt.title(cad)
plt.legend()
plt.show()

# Acercamiento de minimos
a.sort(key=sortFirst)
fig = plt.figure(figsize = (7,5))
ax1 = fig.gca(projection='3d')
ax1.plot_surface(x, h2, C,rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax1.contour(x, h2, C, zdir='z', offset=0, cmap=plt.cm.coolwarm)
ax1.set_xlabel('Carretera(int)')
ax1.set_ylabel('h2')
ax1.set_zlabel('Iluminacion(W)')
ax1.set_xlim(a[0][0],a[len(a)-1][0])
ax1.set_ylim(3,10)
a.sort(key=sortSecond)
ax1.set_zlim(a[0][1],a[len(a)-1][1]+300)
fig.tight_layout()
plt.title(cad)
plt.legend()
plt.show()

# h2 x X
fig = plt.figure(figsize = (7,5))
a.sort(key=sortThird)
plt.title(cad)
plt.plot([sub[2] for sub in a], [sub[0] for sub in a],'--')
plt.xlabel('h2')
plt.ylabel('Carretera(int)') 
plt.grid(b=True, linestyle="--")
plt.show()
