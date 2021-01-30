import matplotlib.pyplot as plt
import numpy as np
from metodo_biseccion import biseccion
import pdb

# d/dx (p_1*h_1)/sqrt((h_1^2+x^2)^3)+(p_2*h_2)/sqrt((h_2^2+(s-x)^2)^3)

# Altura a la cual han sido colocadas las bombillas h1 y h2 respectivamente
h1, h2 = np.random.uniform(low=3, high=10, size=(2,))

# Potencia del bombillo 1 y 2
p1, p2 = np.random.uniform(low=1000, high=4000, size=(2,))

# Distancia horizontal que separa a las bombillas
s = np.random.uniform(low=1, high=30)

# Malla de 200 puntos igualmente separados

x = np.linspace(0, s, 200)

MAXIMO = "max"
MINIMO = "min"
DESCONOCIDO = "unknown"


def L1(x, p, h):
    """Iluminación proveniente de la bombilla 1"""
    res = []
    for xval in x:
        res.append(
            (p * h) / np.sqrt((h**2 + xval**2)**3)
        )
    return res


def L2(x, p, h):
    """Iluminación proveniente de la bombilla 2"""
    res = []
    for xval in x:
        res.append(
            (p * h) / np.sqrt((h**2 + (s - xval)**2)**3)
        )
    return res


def C(x):
    """Iluminación en el punto x sobre la carretera"""
    return np.sum([L1(x, p1, h1), L2(x, p2, h2)], axis=0)


def dx_C(x):
    """Derivada de la función C(x) d[C(x)]/dx"""
    # Primer termino de la derivada
    dxL1 = ((3 * h2 * p2 * (s - x))) / (((s - x)**2) + (h2**2))**(5 / 2)

    dxL2 = (3 * h1 * p1 * x) / (x**2 + h1**2)**(5 / 2)

    return dxL1 - dxL2


def sdx_C(x):
    """Segunda derivada de C(X)"""
    t1 = -(3 * h1 * p1) / ((x**2) + (h1**2))**(5 / 2)
    t2 = (15 * h1 * p1 * (x**2)) / ((x**2) + (h1**2))**(7 / 2)
    t3 = (15 * h2 * p2 * ((s - x)**2)) / (((s - x) ** 2) + (h2**2))**(7 / 2)
    t4 = (3 * h2 * p2) / (((s - x) ** 2) + (h2**2))**(7 / 2)
    return t1 + t2 + t3 + t4


def L1minusL2(x):
    t1 = (h1 * p1) / ((x**2) + (h1**2))**(3 / 2)
    t2 = -(h2 * p2) / (((s - x)**2) + (h2**2))**(3 / 2)
    return t1 + t2


def clasificar(xvalor):
    """ Clasificacion de puntos críticos utilizando el criterio de la segunda derivada"""
    clasificacion = {'xvalor': xvalor}
    if sdx_C(xvalor) > 0:
        clasificacion['tipo'] = MINIMO
        print(f"xvalor = {xvalor} es mínimo local")
    elif sdx_C(xvalor) < 0:
        clasificacion['tipo'] = MAXIMO
        print(f"xvalor = {xvalor} es máximo local")
    else:
        clasificacion['tipo'] = DESCONOCIDO
        print(f"xvalor = {xvalor} es otra cosa")
    return clasificacion


h2 = h1
p2 = p1

# Encontrar aproximaciones iniciales
yL1 = np.array(L1(x, p1, h1))
yL2 = np.array(L2(x, p2, h2))
yC = C(x)
ydxC = [dx_C(i) for i in x]

x0max = x[0]
for xi in x:
    if C([xi]) > C([x0max]):
        x0max = xi
x0min = x[0]
for xi in x:
    if C([xi]) < C([x0min]):
        x0min = xi
x0eq = x[0]
for xi in x:
    if L1minusL2(xi) < L1minusL2(x0eq):
        x0eq = xi

yLdiff = yL1 - yL2
pos = np.where(yLdiff == np.min(yLdiff))
x0eq = x[pos[0][0]]

# Encontrar aproximaciones mejores
xmin = biseccion(x0min - 2, x0min + 2, dx_C, 40, 10e-3)
xmax = biseccion(x0max - 2, x0max + 2, dx_C, 40, 10e-3)
x2max = biseccion(s - 2, s, dx_C, 40, 10e-3)
xeq = biseccion((s / 2) - 4, (s / 2) + 4, L1minusL2, 40, 10e-3)
print(f"x0eq = {x0eq}, xeq={xeq}")
print(f"x0min = {x0min}, x0max={x0max}")

plt.plot(x, L1(x, p1, h1), label='$I_1(x)$')
plt.plot(x, L2(x, p2, h2), label='$I_2(x)$')
plt.plot(x, C(x), label='$C(x)$')

plt.plot(x, [dx_C(i) for i in x], label='$C\'(x)$')

# ==== Graficar puntos críticos ====
if xmin is not None:
    plt.plot(xmin, C([xmin]), 'ro', label='$x_{min}$')
if xmax is not None:
    plt.plot(xmax, C([xmax]), 'bo', label='$x_{max}$')
if xeq is not None:

    plt.plot(xeq, L1([xeq], p1, h1), 'yo', label='$x_{eq}$')
if x2max is not None:
    plt.plot(x2max, C([x2max]), 'bo', label='$x_{2max}$')

plt.xlabel('Distancia que separa a las bombillas')
plt.ylabel('Iluminación')

# Hacer titulo del gráfico
title = f"Gráfica con 3 curvas\n h1={h1}        h2={h2}\n p1={p1}         p2={p2}         s={s}"
plt.title(title)

plt.grid(b=True)
plt.legend()
plt.show()
