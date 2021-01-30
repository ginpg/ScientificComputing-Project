#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Giselt Parra, 26.609.640
# Diego Sanchez, 26.334.929


import numpy as np
import scipy as sp
import random


h1 = random.uniform(3, 10) 
p1 = random.uniform(1000, 4000)
p2 = random.uniform(1000, 4000)
s = random.uniform(0, 30)


cad = "Función de iluminación \n\np1 =" +str(p1)+', h1 = '+str(h1) +' \np2 = '+str(p2)
limiteError = 1e-10

def funL1(x):
    return (p1 * h1) / np.sqrt((h1**2 + x**2)**3)

def funL2(x, h):
    return (p2 * h) / np.sqrt((h**2 + (s - x)**2)**3)

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


def dif(x,h):
    return np.matrix([(3*h*p2*(s-x))/((s-x)**2+h**2)**(5/2)-(3*h1*p1*x)/(x**2+h1**2)**(5/2)
,-(p2*(2*h**2-x**2+2*s*x-s**2))/(h**2+(s-x)**2)**(5/2)]).T

def diff(x,h):
    return np.matrix([[-(3*h1*p1)/(x**2+h1**2)**(5/2)+(15*h1*p1*x**2)/(x**2+h1**2)**(7/2)+(15*h*p2*(s-x)**2)/((s-x)**2+h**2)**(7/2)-(3*h*p2)/((s-x)**2+h**2)**(5/2)
,(3*p2*(x-s)*(4*h**2-x**2+2*s*x-s**2))/(h**2+(s-x)**2)**(7/2)]
,[(3*p2*h*(2*h**2-3*x**2+6*s*x-3*s**2))/(h**2+(s-x)**2)**(7/2)
,-(3*p2*(x-s)*(x**2-2*s*x+s**2-4*h**2))/((s-x)**2+h**2)**(7/2)]])

def BFGS(x,H,tol,maxiter):
    for i in range(0,maxiter):
        #print('\n       ',i)
        #print("x:\n",x)
        p = sp.linalg.solve(H,dif(x.item(0),x.item(1))*-1)
        #print("p:\n", sp.linalg.solve(H,dif(x.item(0),x.item(1))*-1))
        xn = x + p
        #print("xn:\n",xn)
        Cxn = dif(xn.item(0),xn.item(1))
        #print("Cxn:\n",Cxn)
        Cx = dif(x.item(0),x.item(1))
        #print("Cx:\n",Cx)
        y = Cxn - Cx
        #print("y:\n",y)
        H = (H + y*y.T)/(y.T*p) + Cxn*Cx.T/p.T*Cx
        #print("H:\n",H)
        x = xn
        #print("x:\n",x)
        #print("NORMA: ",np.linalg.norm(Cxn))
        if np.linalg.norm(Cxn)<=tol:
            break            
    return xn

def sortFirst(val): 
    return val[0]

def sortSecond(val): 
    return val[1]

def sortThird(val): 
    return val[2]


x = np.linspace(0, s)
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


fig = plt.figure(figsize = (7,5))
ax = fig.gca(projection='3d')
ax.plot_surface(x, h2, C,rstride=1, cstride=1, cmap=plt.cm.coolwarm)
ax.scatter(xminimos, h21d, yminimos, c='m',label = 'puntos mínimos')
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


print("s=",s)
print("p1= ",p1)
print("h1= ",h1)
print("p2= ",p2)

print("\nMinimo con mejor iluminacion: \n("+str(minmax[0])+", "+str(minmax[2])+")")
H = diff(minmax[0],minmax[2])

det = H.item(0,0)*H.item(1,1) - H.item(0,1)*H.item(1,0)

# Vector Gradiente
print("\nVector Gradiente:\n",dif(minmax[0],minmax[2]))

# Matriz Hessiana
print("\nMatriz Hessiana (H):\n",H)

#Minimo, maximo o punto silla
if det < 0:
    print("\ndet(H)= "+str(det)+"\nClasificacion: punto silla")
elif H.item(0,0)<0:
    print("\ndet(H)= "+str(det)+"\nClasificacion: maximo")
elif H.item(0,0)>0:
    print("\ndet(H)= "+str(det)+"\nClasificacion: minimo")
else:
    print("\ndet(H)= "+str(det)+"\nClasificacion:Indefinino")

