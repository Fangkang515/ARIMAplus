"""
This module describes creating and learning neural networks.
"""

import arimaplus_math
import random

# TODO calculations could be done recursively
class simpleNetwork(object):
    """Class defines a neural network of classic neurons, for input of size okno,
    topology specified in [[neurony]] dictionary"""
    def __init__(self, window, neurony, forecast_type, arima):
        self.layers = len(neurony)
        self.topology = [[] for i in range(self.layers)]
        for i in range(0, self.layers):
            quantity = neurony[i]
            for j in range(0, int(quantity)):
                if i == 0:
                    self.topology[i].append(simpleNeuron(window))
                else:
                    self.topology[i].append(simpleNeuron(neurony[i - 1]))
        # append layers of output neurons
        if forecast_type == 1:  # only ann
            self.topology.append([simpleNeuron(neurony[len(neurony) - 1]) for i in range(window)])
        elif forecast_type == 4:  # linear regresion with ann
            self.topology.append([simpleNeuron(neurony[len(neurony) - 1])])
        elif forecast_type == 5:  # polynomial regression with ann
            self.topology.append([simpleNeuron(neurony[len(neurony) - 1]), simpleNeuron(neurony(len(neurony)))])
        elif forecast_type == 6:  # arima with ann (ann forecasts errors)
            self.topology.append([simpleNeuron(neurony[len(neurony) - 1]) for i in range(arima)])
        elif forecast_type == 7:  # arima with ann (ann forecasts errors) (intentionaly duplicated)
            self.topology.append([simpleNeuron(neurony[len(neurony) - 1]) for i in range(arima)])
        self.layers += 1

    def forward_pass(self, input=[]):
        """Executes a single forward pass on the network."""
        self.output = []

        for i in range(self.layers):
            if i == 0:
                for j, el in enumerate(self.topology[i]):
                    self.topology[i][j].calculate(input)
            else:
                for j, el in enumerate(self.topology[i]):
                    self.topology[i][j].calculate([simpleNeuron.output for w in self.topology[i - 1]])
                    if i == self.layers - 1:
                        self.output.append(
                            self.topology[i][j].calculate([simpleNeuron.output for w in self.topology[i - 1]]))
                        # rekurencja bardzo /?
        return self.output

    def backward_pass(self, target, learning_lambda, input=[]):
        """Executes a single learning iteration on the network,
        given target value, learning speed, and input data."""
        print("simpleNetwork bachward pass target:")
        print(target)
        for i in reversed(range(0, self.layers)):
                    if i == self.layers - 1:
                        if not isinstance(target, float):
                            for j, target in enumerate(target):
                                self.topology[i][j].learn(target, learning_lambda, simpleNeuron.getOutputs(self.topology[i-1]))
                        else:
                            self.topology[i][0].learn(target, learning_lambda, simpleNeuron.getOutputs(self.topology[i - 1]))
                    elif i == 0:
                        for j in self.topology[i]:
                            self.topology[i][j].learn(simpleNeuron.getMomentums(j, self.topology[i+1]), learning_lambda, input)
                    else:
                        for j in self.topology[i]:
                            self.topology[i][j].learn(simpleNeuron.getMomentums(j, self.topology[i+1]), learning_lambda, simpleNeuron.getOutputs(self.topology[i-1]))

class simpleNeuron(object):
    """Class defines a single neuron of classic design."""
    output = 0

    def __init__(self, window):
        self.weights = []
        self.sum = 0
        self.bias = 1
        for i in range(0, int(window)):
            self.weights.append(1 / random.randint(1, window))
        self.bias_weight = 1 / random.randint(1, window)

    def calculate(self, input=[]):
        """Calculates output value of *this* neuron, given input
        and using remembered wages."""
        self.sum = 0
        self.sum = sum(x * y for x, y in zip(self.weights, input))
        self.sum += self.bias * self.bias_weight
        output = arimaplus_math.tanh(self.sum)
        return output

    def learn(self, target, learning_lambda, input=[]):
        """Executes single learnig pass."""
        self.d_wagi = []
        if not isinstance(input, int):
            for i in range(len(input)):
                self.d_wagi.append(0)
            for i in range(len(input)):
                self.d_wagi[i] = target * arimaplus_math.derivative_tanh(self.output) * input[i]
            for i in range(len(input)):
                self.weights[i] += learning_lambda * self.d_wagi[i]
            self.bias_weight += target * arimaplus_math.derivative_tanh(self.output) * self.bias * learning_lambda

    def getWeightMomentum(self, n):
        return self.d_wagi[n]

    def getMomentums(self, which_neuron, neurons = []):
        momentums = 0
        for n in neurons:
            momentums += n.getWeightMomentum(which_neuron)
        return momentums

    def getOutputs(self, neurons = []):
        outputs = 0
        for n in neurons:
            outputs += n.output
        return outputs



class lstmNetwork(object):
    """Class defines a neural network of long-short term memory neurons, for input of size okno,
    topology specified in neurony dictionary"""
    def __init__(self, window, neurony, forecast_type, arima):
        self.layers = len(neurony)
        self.topology = [[] for i in range(self.layers)]
        for i in range(self.layers):
            quantity = neurony[i]
            for j in range(0, quantity):
                if i == 0:
                    self.topology[i].append(lstmNeuron(window))
                else:
                    self.topology[i].append(lstmNeuron(neurony[i - 1]))
        # adding output neurons
        if forecast_type == 1:
            self.topology.append([lstmNeuron(neurony(len(neurony))) for i in range(window)])
        elif forecast_type == 4:
            self.topology.append([lstmNeuron(neurony(len(neurony)))])
        elif forecast_type == 5:
            self.topology.append([lstmNeuron(neurony(len(neurony))), lstmNeuron(neurony(len(neurony)))])
        elif forecast_type == 6:
            self.topology.append([lstmNeuron(neurony(len(neurony))) for i in range(arima)])
        elif forecast_type == 7:
            self.topology.append([lstmNeuron(neurony(len(neurony))) for i in range(arima)])
        self.layers += 1

    def forward_pass(self, input=[]):
        """Executes a single forward pass on the network."""
        self.output = []
        for i in range(self.layers):
            if i == 0:
                for j in self.topology[i]:
                    self.topology[i][j].calculate(input)
            else:
                for j in self.topology[i]:
                    self.topology[i][j].calculate([lstmNeuron.wynik for w in self.topology[i - 1]])
                    if i == self.layers - 1:
                        self.output.append(
                            self.topology[i][j].calculate([lstmNeuron.wynik for w in self.topology[i - 1]]))

    def backward_pass(self, target, learning_lambda, input=[]):
        """Executes single learnig pass."""
        for i in reversed(self.layers):
            if i == self.layers - 1:
                for j, cel in enumerate(target):
                    self.topology[i][j].learn(cel, learning_lambda, [lstmNeuron.wynik for w in self.topology[i - 1]])
            elif i == 0:
                for j in self.topology[i]:
                    self.topology[i][j].nauka(sum(map(sum, [lstmNeuron.d_wagi for w in self.topology[i + 1]])),
                                              learning_lambda,
                                              input)
            else:
                self.topology[i][j].nauka(sum(map(sum, [lstmNeuron.d_wagi for w in self.topology[i + 1]])),
                                          learning_lambda,
                                          [lstmNeuron.wynik for w in self.topology[i - 1]])



class lstmNeuron(object):
    """Class defines a single neuron of long-short term memory type."""
    def __init__(self, window):
        self.weights = [[] for i in range(4)]
        self.bias_weights = []
        self.suma_in, self.suma_out, self.suma_mem, self.suma_forget = 0, 0, 0, 0
        self.y_in, self.y_forget, self.state, self.y_out = 0, 0, 0, 0
        self.bias_in, self.bias_out, self.bias_forget, self.bias_mem = 1, 1, 1, 1
        self.mem = 0
        self.output = 0
        for j in range(4):
            for i in range(window):
                self.weights[j][i].append(1 / random.randint(1, 4 * window))
            self.bias_weights.append(1 / random.randint(1, 4 * window))
        self.y_prev = 0
        self.waga_prev = 1 / random.randint(1, 4 * window)

    def calculate(self, input=[]):
        """Executes a single forward pass on a neuron."""
        self.y_prev = self.output
        self.state = self.mem
        # ?? self.stan += self.mem
        self.suma_in = 0
        for i in range(len(input)):
            self.suma_in += input[i] * self.weights[0][i]
        self.suma_in += self.bias_weights[0] * self.bias_in
        self.y_in = arimaplus_math.sigmoid(self.suma_in)

        self.suma_forget = 0
        for i in range(len(input)):
            self.suma_forget += input[i] * self.weights[1][i]
        self.suma_forget += self.bias_weights[1] * self.bias_forget
        self.y_forget = arimaplus_math.sigmoid(self.suma_forget)

        self.suma_mem = 0
        for i in range(len(input)):
            self.suma_mem += input[i] * self.weights[2][i]
        self.suma_mem += self.bias_weights[2] * self.bias_mem
        self.suma_mem += self.y_prev * self.waga_prev
        self.mem = self.y_forget * self.state + self.y_in * arimaplus_math.tanh(self.suma_mem)

        self.suma_out = 0
        for i in range(len(input)):
            self.suma_out += input[i] * self.weights[3][i]
        self.suma_out += self.bias_weights[3] * self.bias_out
        self.y_out = arimaplus_math.sigmoid(self.suma_out)

        self.output = arimaplus_math.tanh(self.mem) * self.y_out

        return self.output

    def learn(self, target, learning_lambda, input=[]):
        """Executes a single learning pass on a neuron."""
        self.d_wagi = [[] for i in range(4)]
        for j in range(4):
            for i in range(len(input)):
                self.d_wagi[j].append(0)

        for i in range(len(input)):
            self.d_wagi[0][i] = target * self.y_out * arimaplus_math.derivative_tanh(
                self.mem) * self.y_forget * arimaplus_math.derivative_tanh(
                self.suma_mem) * arimaplus_math.derivative_sigmoid(self.suma_in) * input[i]
            self.d_wagi[1][i] = target * self.y_out * arimaplus_math.derivative_tanh(
                self.mem) * self.y_in * arimaplus_math.derivative_tanh(self.suma_mem) * arimaplus_math.derivative_sigmoid(
                self.suma_forget) * input[i]
            self.d_wagi[2][i] = target * self.y_out * arimaplus_math.derivative_tanh(
                self.mem) * self.y_forget * self.y_in * arimaplus_math.derivative_tanh(self.suma_mem) * input[i]
            self.d_wagi[3][i] = target * arimaplus_math.derivative_tanh(self.mem) * arimaplus_math.derivative_sigmoid(
                self.suma_out) * input[i]
        self.bias_weights[0] = learning_lambda * target * self.y_out * arimaplus_math.derivative_tanh(
            self.mem) * self.y_forget * arimaplus_math.derivative_tanh(self.suma_mem) * arimaplus_math.derivative_sigmoid(
            self.suma_in) * self.bias_in
        self.bias_weights[1] = learning_lambda * target * self.y_out * arimaplus_math.derivative_tanh(
            self.mem) * self.y_in * arimaplus_math.derivative_tanh(self.suma_mem) * arimaplus_math.derivative_sigmoid(
            self.suma_forget) * self.bias_out
        self.bias_weights[2] = learning_lambda * target * self.y_out * arimaplus_math.derivative_tanh(
            self.mem) * self.y_forget * self.y_in * arimaplus_math.derivative_tanh(self.suma_mem) * self.bias_mem
        self.bias_weights[3] = learning_lambda * target * arimaplus_math.derivative_tanh(self.mem) * arimaplus_math.derivative_sigmoid(
            self.suma_out) * self.bias_forget
        self.waga_prev += learning_lambda * target * self.y_out * arimaplus_math.derivative_tanh(
            self.mem) * self.y_forget * self.y_in * arimaplus_math.derivative_tanh(self.suma_mem) * self.y_prev

        for i in range(len(input)):
            self.weights[0][i] += learning_lambda * self.d_wagi[0][i]
            self.weights[1][i] += learning_lambda * self.d_wagi[1][i]
            self.weights[2][i] += learning_lambda * self.d_wagi[2][i]
            self.weights[3][i] += learning_lambda * self.d_wagi[3][i]


class gruNetwork(object):
    """Class defines a neural network of Gated Recurent Units, for input of size okno,
    topology specified in neurony dictionary"""
    def __init__(self, window, neurony, forecast_type, arima):
        self.layers = len(neurony)
        self.topology = [[] for i in range(self.layers)]
        for i in range(self.layers):
            quantity = neurony[i]
            for j in range(0, quantity):
                if i == 0:
                    self.topology[i].append(gruNeuron(window))
                else:
                    self.topology[i].append(gruNeuron(neurony[i - 1]))
        # adding output neurons
        if forecast_type == 1:
            self.topology.append([gruNeuron(neurony(len(neurony))) for i in range(window)])
        elif forecast_type == 4:
            self.topology.append([gruNeuron(neurony(len(neurony)))])
        elif forecast_type == 5:
            self.topology.append([gruNeuron(neurony(len(neurony))), gruNeuron(neurony(len(neurony)))])
        elif forecast_type == 6:
            self.topology.append([gruNeuron(neurony(len(neurony))) for i in range(arima)])
        elif forecast_type == 7:
            self.topology.append([gruNeuron(neurony(len(neurony))) for i in range(arima)])
        self.layers += 1

    def forward_pass(self, input=[]):
        """Executes a single forward pass on the network."""
        self.wynik = []
        for i in range(self.layers):
            if i == 0:
                for j in self.topology[i]:
                    self.topology[i][j].calculate(input)
            else:
                for j in self.topology[i]:
                    self.topology[i][j].calculate([gruNeuron.wynik for w in self.topology[i - 1]])
                    if i == len(self.layers) - 1:
                        self.wynik.append(
                            self.topology[i][j].calculate([gruNeuron.wynik for w in self.topology[i - 1]]))

    def backward_pass(self, target, learing_lambda, input=[]):
        """Executes single learnig pass."""
        for i in reversed(self.layers):
            if i == len(self.layers) - 1:
                for j, cel in enumerate(target):
                    self.topology[i][j].learn(cel, learing_lambda, [gruNeuron.wynik for w in self.topology[i - 1]])
            elif i == 0:
                for j in self.topology[i]:
                    self.topology[i][j].nauka(sum(map(sum, [gruNeuron.d_wagi for w in self.topology[i + 1]])),
                                              learing_lambda,
                                              input)
            else:
                self.topology[i][j].nauka(sum(map(sum, [gruNeuron.d_wagi for w in self.topology[i + 1]])),
                                          learing_lambda,
                                          [gruNeuron.wynik for w in self.topology[i - 1]])


class gruNeuron(object):
    """Class defines a Gated Recurent Unit."""
    def __init__(self, window):
        self.weights = [[] for i in range(3)]
        self.bias_weight = []
        self.input_weight = []
        self.suma_zt, self.suma_rt, self.suma_we = 0, 0, 0
        self.y_zt, self.y_rt, self.y_we = 0, 0, 0
        self.bias_zt, self.bias_rt, self.bias_we = 1, 1, 1
        self.output = 0
        self.bias = 1

        for j in range(3):
            for i in range(window):
                self.weights[j][i].append(1 / random.randint(1, 3 * window))
            self.bias_weight.append(1 / random.randint(1, 3 * window))
            self.input_weight.append(1 / random.randint(1, 3 * window))
        self.y_prev = 0
        self.input_weight.append(1 / random.randint(1, 4 * window))

    def calculate(self, input=[]):
        """Executes a single forward pass on a neuron."""
        self.y_prev = self.output

        self.suma_zt = 0
        for i in range(len(input)):
            self.suma_zt += self.weights[0][i] * input[i]
        self.suma_zt += self.y_prev * self.input_weight[0]
        self.suma_zt += self.bias_weight[0] * self.bias
        self.y_zt = arimaplus_math.sigmoid(self.suma_zt)

        self.suma_rt = 0
        for i in range(len(input)):
            self.suma_rt += self.weights[1][i] * input[i]
        self.suma_rt += self.y_prev * self.input_weight[1]
        self.suma_rt += self.bias_weight[1] * self.bias
        self.y_rt = arimaplus_math.sigmoid(self.suma_rt)

        self.suma_we = 0
        for i in range(len(input)):
            self.suma_we += self.weights[2][i] * input[i]
        self.suma_we += self.y_prev * self.input_weight[2] * self.y_rt
        self.suma_we += self.bias_weight[2] * self.bias
        self.y_we = arimaplus_math.tanh(self.suma_we)

        self.output = (1 - self.y_zt) * self.y_prev + self.y_zt * self.y_we

        return self.output

    def learn(self, cel, wsp_nauki, wejscie=[]):
        """Executes a single learning pass on a neuron."""
        self.d_wagi = [[] for i in range(3)]
        for j in range(3):
            for i in range(len(wejscie)):
                self.d_wagi[j].append(0)

        for i in range(len(wejscie)):
            self.d_wagi[0][i] = wejscie[i] * (-1 * arimaplus_math.derivative_sigmoid(self.suma_zt)) * (
                self.y_we * cel + self.y_prev * cel)
            self.d_wagi[1][i] = wejscie[i] * arimaplus_math.derivative_sigmoid(
                self.suma_rt) * arimaplus_math.derivative_tanh(self.suma_we) * cel * (1 - self.y_zt)
            self.d_wagi[2][i] = wejscie[i] * arimaplus_math.derivative_tanh(self.suma_we) * cel * (1 - self.y_zt)
        self.input_weight[0] += wsp_nauki * self.y_prev * (-1 * arimaplus_math.derivative_sigmoid(self.suma_zt)) * (
            self.y_we * cel + self.y_prev * cel)
        self.input_weight[1] += wsp_nauki * self.y_prev * arimaplus_math.derivative_sigmoid(
            self.suma_rt) * arimaplus_math.derivative_tanh(self.suma_we) * cel * (1 - self.y_zt)
        self.input_weight[2] += wsp_nauki * self.y_prev * arimaplus_math.derivative_tanh(self.suma_we) * cel * (
            1 - self.y_zt)
        self.bias_weight[0] += wsp_nauki * self.bias_zt * (-1 * arimaplus_math.derivative_sigmoid(self.suma_zt)) * (
            self.y_we * cel + self.y_prev * cel)
        self.bias_weight[1] += wsp_nauki * self.bias_rt * arimaplus_math.derivative_sigmoid(
            self.suma_rt) * arimaplus_math.derivative_tanh(self.suma_we) * cel * (1 - self.y_zt)
        self.bias_weight[2] += wsp_nauki * self.bias_we * arimaplus_math.derivative_tanh(self.suma_we) * cel * (
            1 - self.y_zt)

        for i in range(len(wejscie)):
            self.weights[0][i] += wsp_nauki * self.d_wagi[0][i]
            self.weights[1][i] += wsp_nauki * self.d_wagi[1][i]
            self.weights[2][i] += wsp_nauki * self.d_wagi[2][i]
