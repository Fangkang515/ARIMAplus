"""
This module contains prediction algorithms, used mainly
as base to calculating loss functions.
"""

import arimaplus_math as ap_math


def linearRegression(window, input=[]):
    """For a given list of numbers and a window frame,
    slides the frame over sublist of input data
    and returns linear regression coeficient - one for each frame."""
    baza = [n for n in range(0, int(window))]
    output = []
    for i in range(0, int(len(input) / window) - 1):
        output.append(ap_math.linearRegression(baza, input[int(i * window):int(i * window + window)]))

    return output


def polynomialRegression(window, n, input=[]):
    """For a given list of numbers and a window frame,
    slides the frame over sublist of input data
    and returns polynomial regression coeficients for each frame."""
    baza = [n for n in range(0, len(window))]
    output = []
    for i in range(0, len(input / window) - 1):
        output.append(ap_math.polynomialRegression(baza, input[i * window:i * window + window], n))

    return output

# TODO
def DataDifferentiation(q, input=[]):
    """For a given list of numbers creates
    a corresponding list of differenced data,
    making data stationary (I in ARIMA)."""
    output = input
    print("DataDifferentiation for q=" + str(q) + " input: ")
    print(input)
    if q == 1:
        for i in range(1, len(output)):
            output[i] = input[i] - input[i - 1]
    if q == 2:
        output[0] = 0
        output[1] = 0
        for i in range(2, len(output)):
            output[i] = input[i] - input[i - 1]
    if q == 3:
        for i in range(3, len(output)):
            output[i] = input[i] - input[i - 1]

    output = ap_math.normalise(output)

    return output

# TODO
def interpretDifferentiation(q, wejscie=[]):
    """Inverts the differencing process."""
    pass


def ARIMA(p, d, cA, cB, cC, eA, eB, eC, input=[]):
    """Predicts data series values using ARIMA(p, q, d) model.
    Returns list of pairs: output value, error."""
    formula = ""
    output = []
    error = [0, 0, 0]
    average = ap_math.average(input)

    formula += "average"
    if p == 1:
        formula += "+ cA*x"
    if p == 2:
        formula += "+ cA*x+cB*y"
    if p == 3:
        formula += "+ cA*x+cB*y+cC*z"
    if d == 1:
        formula += "-eA*m"
    if d == 2:
        formula += "-eA*m -eB*n"
    if d == 3:
        formula += "-eA*m -eB*n -eC*r"

    print("ARIMA(" + str(p) + ", " + str(d) + ") formula for (" + " " + str(cA) + " " + str(cB) + " " + str(cC) + " " + str(eA) + " " + str(eB) + " " + str(eC) + "): " + formula)
    print("ARIMA(" + str(p) + ", " + str(d) + ") input data with average=" + str(average) + ": ")
    print(input)

    for i in range(3, len(input)):
        x = input[i - 1]
        y = input[i - 2]
        z = input[i - 3]
        m = 0
        n = 0
        p = 0
        if i >= 1:
            m = error[i - 1]
        elif i >= 2:
            n = error[i - 2]
        elif i >= 3:
            r = error[i - 3]
        output.append(eval(formula))
        if i < len(input):
            error.append(input[i] - eval(formula))
    return zip(output, error)
