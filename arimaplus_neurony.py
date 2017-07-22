"""
This module describes creating and learning neural networks.
"""

import arimaplus_math
import random


class siec_zwykly(object):
    """Class defines a neural network of classic neurons, for input of size okno,
    topology specified in neurony dictionary"""
    def __init__(self, okno, neurony, typ_przewidywania, arima):
        self.warstwy = len(neurony)
        self.topologia = [[] for i in range(self.warstwy)]
        for i in range(self.warstwy):
            liczba = neurony[i]
            for j in range(0, int(liczba)):
                if i == 0:
                    self.topologia[i].append(neuron_zwykly(okno))
                else:
                    self.topologia[i].append(neuron_zwykly(neurony[i - 1]))
        # dodanie warstwy neuronów wynikowych
        if typ_przewidywania == 1:  # czysta ann
            self.topologia.append([neuron_zwykly(neurony[len(neurony) - 1]) for i in range(okno)])
        elif typ_przewidywania == 4:  # regresja liniowa z ann
            self.topologia.append([neuron_zwykly(neurony[len(neurony) - 1])])
        elif typ_przewidywania == 5:  # regresja wielomianowa z ann
            self.topologia.append([neuron_zwykly(neurony[len(neurony) - 1]), neuron_zwykly(neurony(len(neurony)))])
        elif typ_przewidywania == 6:  # arima z ann (ann przewiduje błąd)
            self.topologia.append([neuron_zwykly(neurony[len(neurony) - 1]) for i in range(arima)])
        elif typ_przewidywania == 7:  # arima z ann (ann przewiduje błąd)
            self.topologia.append([neuron_zwykly(neurony[len(neurony) - 1]) for i in range(arima)])
        self.warstwy += 1

    def forward_pass(self, wejscie=[]):
        """Executes a single forward pass on the network."""
        self.wynik = []
        for i in range(self.warstwy):
            if i == 0:
                for j, el in enumerate(self.topologia[i]):
                    self.topologia[i][j].przelicz(wejscie)
            else:
                for j, el in enumerate(self.topologia[i]):
                    self.topologia[i][j].przelicz([neuron_zwykly.wynik for w in self.topologia[i - 1]])
                    if i == self.warstwy - 1:
                        self.wynik.append(
                            self.topologia[i][j].przelicz([neuron_zwykly.wynik for w in self.topologia[i - 1]]))
                        # rekurencja bardzo /?
        return self.wynik

    def backward_pass(self, target, wsp_nauki, wejscie=[]):
        """Executes a single learning iteration on the network,
        given target value, learning speed, and input data."""
        for i in reversed(range(0, self.warstwy)):
            #if type(target) is not float:
            try:
                if i == self.warstwy - 1:
                    for j, cel in enumerate(target):
                        self.topologia[i][j].nauka(cel, wsp_nauki, [neuron_zwykly.wynik for w in self.topologia[i - 1]])
                elif i == 0:
                    for j in self.topologia[i]:
                        self.topologia[i][j].nauka(sum([neuron_zwykly.d_wagi for w in self.topologia[i + 1]]), wsp_nauki,
                                                   wejscie)
                else:
                    self.topologia[i][j].nauka(sum([neuron_zwykly.d_wagi for w in self.topologia[i + 1]]), wsp_nauki,
                                               [neuron_zwykly.wynik for w in self.topologia[i - 1]])
            except TypeError:
                if i == self.warstwy - 1:
                    self.topologia[i][0].nauka(cel, wsp_nauki, [neuron_zwykly.wynik for w in self.topologia[i - 1]])
                elif i == 0:
                    for j in self.topologia[i]:
                        self.topologia[i][j].nauka(sum([neuron_zwykly.d_wagi for w in self.topologia[i + 1]]),
                                                   wsp_nauki,
                                                   wejscie)
                else:
                    self.topologia[i][j].nauka(sum([neuron_zwykly.d_wagi for w in self.topologia[i + 1]]), wsp_nauki,
                                               [neuron_zwykly.wynik for w in self.topologia[i - 1]])


class neuron_zwykly(object):
    """Class defines a single neuron of classic design."""
    wynik = 0

    def __init__(self, okno):
        self.wagi = []
        self.suma = 0
        self.bias = 1
        for i in range(0, int(okno)):
            self.wagi.append(1 / random.randint(1, okno))
        self.waga_bias = 1 / random.randint(1, okno)

    def przelicz(self, wejscie=[]):
        """Calculates output value of *this* neuron, given input
        and using remembered wages."""
        self.suma = 0
        self.suma = sum(x * y for x, y in zip(self.wagi, wejscie))
        self.suma += self.bias * self.waga_bias
        wynik = arimaplus_math.tanh(self.suma)
        return wynik

    def nauka(self, cel, wsp_nauki, wejscie=[]):
        """Executes single learnig pass."""
        self.d_wagi = []
        for i in range(len(wejscie)):
            self.d_wagi.append(0)
        for i in range(len(wejscie)):
            self.d_wagi[i] = cel * arimaplus_math.pochodna_tanh(self.wynik) * wejscie[i]
        for i in range(len(wejscie)):
            self.wagi[i] += wsp_nauki * self.d_wagi[i]
        self.waga_bias += cel * arimaplus_math.pochodna_tanh(self.wynik) * self.bias * wsp_nauki


class siec_LSTM(object):
    """Class defines a neural network of long-short term memory neurons, for input of size okno,
    topology specified in neurony dictionary"""
    def __init__(self, okno, neurony, typ_przewidywania, arima):
        self.warstwy = len(neurony)
        self.topologia = [[] for i in range(self.warstwy)]
        for i in range(self.warstwy):
            liczba = neurony[i]
            for j in range(0, liczba):
                if i == 0:
                    self.topologia[i].append(neuron_LSTM(okno))
                else:
                    self.topologia[i].append(neuron_LSTM(neurony[i - 1]))
        # dodanie warstwy neuronów wynikowych
        if typ_przewidywania == 1:
            self.topologia.append([neuron_LSTM(neurony(len(neurony))) for i in range(okno)])
        elif typ_przewidywania == 4:
            self.topologia.append([neuron_LSTM(neurony(len(neurony)))])
        elif typ_przewidywania == 5:
            self.topologia.append([neuron_LSTM(neurony(len(neurony))), neuron_LSTM(neurony(len(neurony)))])
        elif typ_przewidywania == 6:
            self.topologia.append([neuron_LSTM(neurony(len(neurony))) for i in range(arima)])
        elif typ_przewidywania == 7:
            self.topologia.append([neuron_LSTM(neurony(len(neurony))) for i in range(arima)])
        self.warstwy += 1

    def forward_pass(self, wejscie=[]):
        """Executes a single forward pass on the network."""
        self.wynik = []
        for i in range(self.warstwy):
            if i == 0:
                for j in self.topologia[i]:
                    self.topologia[i][j].przelicz(wejscie)
            else:
                for j in self.topologia[i]:
                    self.topologia[i][j].przelicz([neuron_LSTM.wynik for w in self.topologia[i - 1]])
                    if i == self.warstwy - 1:
                        self.wynik.append(
                            self.topologia[i][j].przelicz([neuron_LSTM.wynik for w in self.topologia[i - 1]]))

    def backward_pass(self, target, wsp_nauki, wejscie=[]):
        """Executes single learnig pass."""
        for i in reversed(self.warstwy):
            if i == self.warstwy - 1:
                for j, cel in enumerate(target):
                    self.topologia[i][j].nauka(cel, wsp_nauki, [neuron_LSTM.wynik for w in self.topologia[i - 1]])
            elif i == 0:
                for j in self.topologia[i]:
                    self.topologia[i][j].nauka(sum(map(sum, [neuron_LSTM.d_wagi for w in self.topologia[i + 1]])),
                                               wsp_nauki,
                                               wejscie)
            else:
                self.topologia[i][j].nauka(sum(map(sum, [neuron_LSTM.d_wagi for w in self.topologia[i + 1]])),
                                           wsp_nauki,
                                           [neuron_LSTM.wynik for w in self.topologia[i - 1]])


class neuron_LSTM(object):
    """Class defines a single neuron of long-short term memory type."""
    def __init__(self, okno):
        self.wagi = [[] for i in range(4)]
        self.waga_bias = []
        self.suma_in, self.suma_out, self.suma_mem, self.suma_forget = 0, 0, 0, 0
        self.y_in, self.y_forget, self.stan, self.y_out = 0, 0, 0, 0
        self.bias_in, self.bias_out, self.bias_forget, self.bias_mem = 1, 1, 1, 1
        self.mem = 0
        self.wynik = 0
        for j in range(4):
            for i in range(okno):
                self.wagi[j][i].append(1 / random.randint(1, 4 * okno))
            self.waga_bias.append(1 / random.randint(1, 4 * okno))
        self.y_prev = 0
        self.waga_prev = 1 / random.randint(1, 4 * okno)

    def przelicz(self, wejscie=[]):
        """Executes a single forward pass on a neuron."""
        self.y_prev = self.wynik
        self.stan = self.mem
        # ?? self.stan += self.mem
        self.suma_in = 0
        for i in range(len(wejscie)):
            self.suma_in += wejscie[i] * self.wagi[0][i]
        self.suma_in += self.waga_bias[0] * self.bias_in
        self.y_in = arimaplus_math.sigmoid(self.suma_in)

        self.suma_forget = 0
        for i in range(len(wejscie)):
            self.suma_forget += wejscie[i] * self.wagi[1][i]
        self.suma_forget += self.waga_bias[1] * self.bias_forget
        self.y_forget = arimaplus_math.sigmoid(self.suma_forget)

        self.suma_mem = 0
        for i in range(len(wejscie)):
            self.suma_mem += wejscie[i] * self.wagi[2][i]
        self.suma_mem += self.waga_bias[2] * self.bias_mem
        self.suma_mem += self.y_prev * self.waga_prev
        self.mem = self.y_forget * self.stan + self.y_in * arimaplus_math.tanh(self.suma_mem)

        self.suma_out = 0
        for i in range(len(wejscie)):
            self.suma_out += wejscie[i] * self.wagi[3][i]
        self.suma_out += self.waga_bias[3] * self.bias_out
        self.y_out = arimaplus_math.sigmoid(self.suma_out)

        self.wynik = arimaplus_math.tanh(self.mem) * self.y_out

        return self.wynik

    def nauka(self, cel, wsp_nauki, wejscie=[]):
        """Executes a single learning pass on a neuron."""
        self.d_wagi = [[] for i in range(4)]
        for j in range(4):
            for i in range(len(wejscie)):
                self.d_wagi[j].append(0)

        for i in range(len(wejscie)):
            self.d_wagi[0][i] = cel * self.y_out * arimaplus_math.pochodna_tanh(
                self.mem) * self.y_forget * arimaplus_math.pochodna_tanh(
                self.suma_mem) * arimaplus_math.pochodna_sigmoid(self.suma_in) * wejscie[i]
            self.d_wagi[1][i] = cel * self.y_out * arimaplus_math.pochodna_tanh(
                self.mem) * self.y_in * arimaplus_math.pochodna_tanh(self.suma_mem) * arimaplus_math.pochodna_sigmoid(
                self.suma_forget) * wejscie[i]
            self.d_wagi[2][i] = cel * self.y_out * arimaplus_math.pochodna_tanh(
                self.mem) * self.y_forget * self.y_in * arimaplus_math.pochodna_tanh(self.suma_mem) * wejscie[i]
            self.d_wagi[3][i] = cel * arimaplus_math.pochodna_tanh(self.mem) * arimaplus_math.pochodna_sigmoid(
                self.suma_out) * wejscie[i]
        self.waga_bias[0] = wsp_nauki * cel * self.y_out * arimaplus_math.pochodna_tanh(
            self.mem) * self.y_forget * arimaplus_math.pochodna_tanh(self.suma_mem) * arimaplus_math.pochodna_sigmoid(
            self.suma_in) * self.bias_in
        self.waga_bias[1] = wsp_nauki * cel * self.y_out * arimaplus_math.pochodna_tanh(
            self.mem) * self.y_in * arimaplus_math.pochodna_tanh(self.suma_mem) * arimaplus_math.pochodna_sigmoid(
            self.suma_forget) * self.bias_out
        self.waga_bias[2] = wsp_nauki * cel * self.y_out * arimaplus_math.pochodna_tanh(
            self.mem) * self.y_forget * self.y_in * arimaplus_math.pochodna_tanh(self.suma_mem) * self.bias_mem
        self.waga_bias[3] = wsp_nauki * cel * arimaplus_math.pochodna_tanh(self.mem) * arimaplus_math.pochodna_sigmoid(
            self.suma_out) * self.bias_forget
        self.waga_prev += wsp_nauki * cel * self.y_out * arimaplus_math.pochodna_tanh(
            self.mem) * self.y_forget * self.y_in * arimaplus_math.pochodna_tanh(self.suma_mem) * self.y_prev

        for i in range(len(wejscie)):
            self.wagi[0][i] += wsp_nauki * self.d_wagi[0][i]
            self.wagi[1][i] += wsp_nauki * self.d_wagi[1][i]
            self.wagi[2][i] += wsp_nauki * self.d_wagi[2][i]
            self.wagi[3][i] += wsp_nauki * self.d_wagi[3][i]


class siec_GRU(object):
    """Class defines a neural network of Gated Recurent Units, for input of size okno,
    topology specified in neurony dictionary"""
    def __init__(self, okno, neurony, typ_przewidywania, arima):
        self.warstwy = len(neurony)
        self.topologia = [[] for i in range(self.warstwy)]
        for i in range(self.warstwy):
            liczba = neurony[i]
            for j in range(0, liczba):
                if i == 0:
                    self.topologia[i].append(neuron_GRU(okno))
                else:
                    self.topologia[i].append(neuron_GRU(neurony[i - 1]))
        # dodanie warstwy neuronów wynikowych
        if typ_przewidywania == 1:
            self.topologia.append([neuron_GRU(neurony(len(neurony))) for i in range(okno)])
        elif typ_przewidywania == 4:
            self.topologia.append([neuron_GRU(neurony(len(neurony)))])
        elif typ_przewidywania == 5:
            self.topologia.append([neuron_GRU(neurony(len(neurony))), neuron_GRU(neurony(len(neurony)))])
        elif typ_przewidywania == 6:
            self.topologia.append([neuron_GRU(neurony(len(neurony))) for i in range(arima)])
        elif typ_przewidywania == 7:
            self.topologia.append([neuron_GRU(neurony(len(neurony))) for i in range(arima)])
        self.warstwy += 1

    def forward_pass(self, wejscie=[]):
        """Executes a single forward pass on the network."""
        self.wynik = []
        for i in range(self.warstwy):
            if i == 0:
                for j in self.topologia[i]:
                    self.topologia[i][j].przelicz(wejscie)
            else:
                for j in self.topologia[i]:
                    self.topologia[i][j].przelicz([neuron_GRU.wynik for w in self.topologia[i - 1]])
                    if i == len(self.warstwy) - 1:
                        self.wynik.append(
                            self.topologia[i][j].przelicz([neuron_GRU.wynik for w in self.topologia[i - 1]]))

    def backward_pass(self, target, wsp_nauki, wejscie=[]):
        """Executes single learnig pass."""
        for i in reversed(self.warstwy):
            if i == len(self.warstwy) - 1:
                for j, cel in enumerate(target):
                    self.topologia[i][j].nauka(cel, wsp_nauki, [neuron_GRU.wynik for w in self.topologia[i - 1]])
            elif i == 0:
                for j in self.topologia[i]:
                    self.topologia[i][j].nauka(sum(map(sum, [neuron_GRU.d_wagi for w in self.topologia[i + 1]])),
                                               wsp_nauki,
                                               wejscie)
            else:
                self.topologia[i][j].nauka(sum(map(sum, [neuron_GRU.d_wagi for w in self.topologia[i + 1]])),
                                           wsp_nauki,
                                           [neuron_GRU.wynik for w in self.topologia[i - 1]])


class neuron_GRU(object):
    """Class defines a Gated Recurent Unit."""
    def __init__(self, okno):
        self.wagi = [[] for i in range(3)]
        self.waga_bias = []
        self.waga_prev = []
        self.suma_zt, self.suma_rt, self.suma_we = 0, 0, 0
        self.y_zt, self.y_rt, self.y_we = 0, 0, 0
        self.bias_zt, self.bias_rt, self.bias_we = 1, 1, 1
        self.wynik = 0
        self.bias = 1

        for j in range(3):
            for i in range(okno):
                self.wagi[j][i].append(1 / random.randint(1, 3 * okno))
            self.waga_bias.append(1 / random.randint(1, 3 * okno))
            self.waga_prev.append(1 / random.randint(1, 3 * okno))
        self.y_prev = 0
        self.waga_prev.append(1 / random.randint(1, 4 * okno))

    def przelicz(self, wejscie=[]):
        """Executes a single forward pass on a neuron."""
        self.y_prev = self.wynik

        self.suma_zt = 0
        for i in range(len(wejscie)):
            self.suma_zt += self.wagi[0][i] * wejscie[i]
        self.suma_zt += self.y_prev * self.waga_prev[0]
        self.suma_zt += self.waga_bias[0] * self.bias
        self.y_zt = arimaplus_math.sigmoid(self.suma_zt)

        self.suma_rt = 0
        for i in range(len(wejscie)):
            self.suma_rt += self.wagi[1][i] * wejscie[i]
        self.suma_rt += self.y_prev * self.waga_prev[1]
        self.suma_rt += self.waga_bias[1] * self.bias
        self.y_rt = arimaplus_math.sigmoid(self.suma_rt)

        self.suma_we = 0
        for i in range(len(wejscie)):
            self.suma_we += self.wagi[2][i] * wejscie[i]
        self.suma_we += self.y_prev * self.waga_prev[2] * self.y_rt
        self.suma_we += self.waga_bias[2] * self.bias
        self.y_we = arimaplus_math.tanh(self.suma_we)

        self.wynik = (1 - self.y_zt) * self.y_prev + self.y_zt * self.y_we

        return self.wynik

    def nauka(self, cel, wsp_nauki, wejscie=[]):
        """Executes a single learning pass on a neuron."""
        self.d_wagi = [[] for i in range(3)]
        for j in range(3):
            for i in range(len(wejscie)):
                self.d_wagi[j].append(0)

        for i in range(len(wejscie)):
            self.d_wagi[0][i] = wejscie[i] * (-1 * arimaplus_math.pochodna_sigmoid(self.suma_zt)) * (
                self.y_we * cel + self.y_prev * cel)
            self.d_wagi[1][i] = wejscie[i] * arimaplus_math.pochodna_sigmoid(
                self.suma_rt) * arimaplus_math.pochodna_tanh(self.suma_we) * cel * (1 - self.y_zt)
            self.d_wagi[2][i] = wejscie[i] * arimaplus_math.pochodna_tanh(self.suma_we) * cel * (1 - self.y_zt)
        self.waga_prev[0] += wsp_nauki * self.y_prev * (-1 * arimaplus_math.pochodna_sigmoid(self.suma_zt)) * (
            self.y_we * cel + self.y_prev * cel)
        self.waga_prev[1] += wsp_nauki * self.y_prev * arimaplus_math.pochodna_sigmoid(
            self.suma_rt) * arimaplus_math.pochodna_tanh(self.suma_we) * cel * (1 - self.y_zt)
        self.waga_prev[2] += wsp_nauki * self.y_prev * arimaplus_math.pochodna_tanh(self.suma_we) * cel * (
            1 - self.y_zt)
        self.waga_bias[0] += wsp_nauki * self.bias_zt * (-1 * arimaplus_math.pochodna_sigmoid(self.suma_zt)) * (
            self.y_we * cel + self.y_prev * cel)
        self.waga_bias[1] += wsp_nauki * self.bias_rt * arimaplus_math.pochodna_sigmoid(
            self.suma_rt) * arimaplus_math.pochodna_tanh(self.suma_we) * cel * (1 - self.y_zt)
        self.waga_bias[2] += wsp_nauki * self.bias_we * arimaplus_math.pochodna_tanh(self.suma_we) * cel * (
            1 - self.y_zt)

        for i in range(len(wejscie)):
            self.wagi[0][i] += wsp_nauki * self.d_wagi[0][i]
            self.wagi[1][i] += wsp_nauki * self.d_wagi[1][i]
            self.wagi[2][i] += wsp_nauki * self.d_wagi[2][i]
