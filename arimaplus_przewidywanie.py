"""
This module contains prediction algorithms, used mainly
as base to calculating loss functions.
"""

import arimaplus_math as ap_math


def licz_reg_liniowa(okno, wejscie=[]):
    baza = [n for n in range(0, int(okno))]
    output = []
    for i in range(0, int(len(wejscie) / okno) - 1):
        output.append(ap_math.regresja_liniowa(baza, wejscie[int(i * okno):int(i * okno + okno)]))

    return output


def licz_reg_wielomianiowa(okno, n, wejscie=[]):
    baza = [n for n in range(0, len(okno))]
    output = []
    for i in range(0, len(wejscie / okno) - 1):
        output.append(ap_math.regresja_wielomianowa(baza, wejscie[i * okno:i * okno + okno], n))

    return output


def roznicoj_dane(q, wejscie=[]):
    output = wejscie
    if q == 1:
        for i in range(1, len(output)):
            output[i] = wejscie[i] - wejscie[i - 1]
    if q == 2:
        for i in range(2, len(output)):
            output[i] = wejscie[i] - 2 * wejscie[i - 1] + wejscie[i - 2]
    if q == 3:
        for i in range(3, len(output)):
            output[i] = wejscie[i] - 2 * wejscie[i - 1] + wejscie[i - 3]

    output = ap_math.normalizuj(output)

    return output


def interpretuj_roznicowanie(q, wejscie=[]):
    pass


def ARIMA(p, d, cA, cB, cC, eA, eB, eC, wejscie=[]):
    rownanie = ""
    output = []
    error = []
    srednia = ap_math.srednia(wejscie)

    rownanie += "srednia"
    if p == 1:
        rownanie += "+ cA*x"
    if p == 2:
        rownanie += "+ cA*x+cB*y"
    if p == 3:
        rownanie += "+ cA*x+cB*y+cC*z"
    if d == 1:
        rownanie += "-eA*m"
    if d == 2:
        rownanie += "-eA*m -eB*n"
    if d == 3:
        rownanie += "-eA*m -eB*n -eC*r"

    for i in range(0, len(wejscie)):
        x = wejscie[i - 1]
        y = wejscie[i - 2]
        z = wejscie[i - 3]
        m = 0
        n = 0
        p = 0
        if i >= 1:
            m = error[i - 1]
        elif i >= 2:
            n = error[i - 2]
        elif i >= 3:
            r = error[i - 3]
        output.append(eval(rownanie))
        if i < len(wejscie):
            error.append(wejscie[i] - eval(rownanie))
    return zip(output, error)
