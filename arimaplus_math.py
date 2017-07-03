"""
This module contains basic mathematic functions.
"""

import math


def srednia(x=[]):
    suma = 0.0
    for el in x:
        suma += el
    return suma / len(x)


def minimum(x=[]):
    wynik = x[0]
    for i in range(len(x)):
        wynik = x[i] if x[i] < wynik else wynik
    return wynik


def maksimum(x=[]):
    wynik = x[0]
    for i in range(len(x)):
        wynik = x[i] if x[i] > wynik else wynik
    return wynik


def normalizuj(x=[]):
    min = minimum(x)
    maks = maksimum(x)
    return [(n - min) / (maks - min) for n in x]


def odchylenie(x=[]):
    mem = srednia(x)
    return (1 / (len(x) - 1)) * math.sqrt(sum((el - mem) * (el - mem) for el in x))


def korelacja(x=[], y=[]):
    memx = srednia(x)
    memy = srednia(y)
    return sum((elx - memx) * (ely - memy) for elx, ely in zip(x, y)) / odchylenie(y) * odchylenie(x) * (
        1 / (len(x) - 1))


def regresja_liniowa(x=[], y=[]):
    return korelacja(x, y) * odchylenie(y) / odchylenie(x)


def wyznacznik(M, n):
    """determinant calculated using LU decomposition"""
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


def regresja_wielomianowa(x, y, n):
    """n ~ order of polynomial
    a_n*x^n + ... + a_0*x"""
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
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(x, y)]) / len(x))


def tanh(x):
    return 2 / (1 + math.exp(-2 * x)) - 1


def pochodna_tanh(x):
    mem = tanh(x)
    return 1 - (mem * mem)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def pochodna_sigmoid(x):
    mem = sigmoid(x)
    return mem * (1 - mem)


def bent(x):
    return (math.sqrt(x * x + 1) - 1) / 2 + x


def pochodna_bent(x):
    return x / (2 * math.sqrt(x * x + 1)) + 1
