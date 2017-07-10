"""
This module contains basic mathematic functions.
"""

import math


def srednia(x=[]):
    """Given a list of numbers, returns their mean value."""
    suma = 0.0
    for el in x:
        suma += el
    return suma / len(x)


def minimum(x=[]):
    """Returns a minimum value from a list of numbers."""
    wynik = x[0]
    for i in range(len(x)):
        wynik = x[i] if x[i] < wynik else wynik
    return wynik


def maksimum(x=[]):
    """Returns a maximum value from a list of numbers."""
    wynik = x[0]
    for i in range(len(x)):
        wynik = x[i] if x[i] > wynik else wynik
    return wynik


def normalizuj(x=[]):
    """Given a list of numbers, returns a normalized <0 1> list."""
    min = minimum(x)
    maks = maksimum(x)
    return [(n - min) / (maks - min) for n in x]


def odchylenie(x=[]):
    """Returns standard deviation for a given list of numbers."""
    mem = srednia(x)
    return (1 / (len(x) - 1)) * math.sqrt(sum((el - mem) * (el - mem) for el in x))


def korelacja(x=[], y=[]):
    """Returns Pearson's product-moment coefficient for two list of numbers."""
    memx = srednia(x)
    memy = srednia(y)
    return sum((elx - memx) * (ely - memy) for elx, ely in zip(x, y)) / odchylenie(y) * odchylenie(x) * (
        1 / (len(x) - 1))


def regresja_liniowa(x=[], y=[]):
    """Returns aproximated a0 coeficient for given two lists of numvers,
    where a0 fits the y = a0*x+a1 model.  """

    return korelacja(x, y) * odchylenie(y) / odchylenie(x)


def wyznacznik(M, n):
    """Matrix determinant calculated using LU decomposition, for given two-dimensional square matrix
    M with order of n. """
    L = [[] for i in range(n)]
    U = [[] for i in range(n)]
    for i in range(n):
        U[i][i] = 1
    for j in range(n):
        for i in range(j, n):
            suma = 0
            for k in range(j):
                suma += L[i][k] * U[k][j]
            L[i][j] = M[i][j] - suma
        for i in range(j, n):
            suma = 0
            for k in range(j):
                suma += L[i][k] * U[k][i]
            U[j][i] = (M[j][i] - suma) / L[j][j]
    detL = 1
    for i in range(n):
        detL *= L[i][i]
    detU = 1
    for i in range(n):
        detU *= U[i][i]
    return detU * detL


def regresja_wielomianowa(x = [], y = [], n = 2):
    """Given lists of numbers and desired degree of polynomial,
    returns a list of coefficients of aproximated polynomial."""
    wynik = []
    M = [[] for i in range(n)]
    My = []
    Mx = [[[] for i in range(n)] for j in range(n)]
    for i in range(n):
        for j in range(n):
            if i == 0 and j == 0:
                M[0].append(len(x))
            else:
                M[i].append(sum(el ** (i + j) for el in x))
    for i in range(n):
        My.append(sum((elx ** i) * ely for elx, ely in zip(x, y)))
    for i in range(n):
        Mx[i] = M
    for i in range(n):
        Mx[i][i] = My
    for i in range(n):
        wynik.append(wyznacznik(Mx[i], n))
    wynik.reverse()
    return wynik


def rmse(x=[], y=[]):
    """For two lists of numbers, returns root-mean-square error of how
    different are coresponding elements of the lists."""
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]) / len(x))


def tanh(x):
    """Returns a value of hyperbolic tangent for a given number."""
    return 2 / (1 + math.exp(-2 * x)) - 1


def pochodna_tanh(x):
    """Returns a value of derivative of hyperbolic tangent for a given number."""
    mem = tanh(x)
    return 1 - (mem * mem)


def sigmoid(x):
    """Returns a value of sigmoid function for a given number."""
    return 1 / (1 + math.exp(-x))


def pochodna_sigmoid(x):
    """Returns a value of derivative of sigmoid function for a given number."""
    mem = sigmoid(x)
    return mem * (1 - mem)


def bent(x):
    """Returns a value of bent identity function for a given number."""
    return (math.sqrt(x * x + 1) - 1) / 2 + x


def pochodna_bent(x):
    """Returns a value of derivative of bent identity for a given number."""
    return x / (2 * math.sqrt(x * x + 1)) + 1
