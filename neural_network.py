import numpy as np


class NeuralNetwork:
    def __init__(self, layers_num, functions):
        self.layers = []
        self.biases = []
        if hasattr(functions, '__iter__') or hasattr(functions, '__getitem__'):
            assert len(layers_num) - 1 == len(functions)
        else:
            functions = (functions,)*(len(layers_num) - 1)
        self.nonlin_funcs = functions
        layers = list(layers_num)
        for i in xrange(len(layers)-1):
            self.layers.append(np.random.rand(layers[i], layers[i+1])-0.5)
            self.biases.append(np.random.rand(layers[i+1])-0.5)

    def hypot(self, input):
        for i in range(len(self.layers)):
            input = self.nonlin_funcs[i](input.dot(self.layers[i])+self.biases[i])
        return input

    def cost(self, input, label):
        return np.sum((self.hypot(input)-label)**2)

    def iteration(self, input, label, alpha=0.01):
        inputs = []
        deltas = []

        for i in xrange(len(self.layers)):
            inputs.append(input)
            input = self.nonlin_funcs[i](input.dot(self.layers[i])+self.biases[i])

        error = label-input
        deltas.append(error*self.nonlin_funcs[-1](input, True))

        for i in range(len(inputs)-1, 0, -1):
            tmpdelta = deltas[-1].dot(self.layers[i].T)*self.nonlin_funcs[i-1](inputs[i], True)
            deltas.append(tmpdelta)

        deltas.reverse()
        inputs.append(input)

        for i in range(len(self.layers)):
            der = inputs[i].T.dot(deltas[i])
            self.layers[i] = self.layers[i] + alpha*der/np.sqrt(10.0 + der**2)/len(inputs[0])
            t_del = deltas[i].T
            for j in range(len(self.biases[i])):
                der = sum(t_del[j])
                self.biases[i][j] += alpha*der/np.sqrt(10.0 + der**2)

        return deltas[0].dot(self.layers[0].T)  # last delta (partial)

    def iteration_by_next_net(self, inputs, delta, alpha=0.01):
        for i in xrange(len(self.layers)-1, -1, -1):
            delta *= self.nonlin_funcs[i](inputs[i+1], True)
            self.layers[i] -= inputs[i].T.dot(delta)
            delta = delta.dot(self.layers[i].T)
        return delta  # last delta (partial)

    def __call__(self, input):
        inp = [input]
        for i in xrange(len(self.layers)):
            inp.append(self.nonlin_funcs[i](inp[-1].dot(self.layers[i])+self.biases[i]))
        return inp

    def deltas_by_next_net(self, inputs, delta):
        deltas = []
        for i in xrange(len(self.layers)-1, -1, -1):
            delta *= self.nonlin_funcs[i](inputs[i+1], True)
            deltas.append(delta)
            delta = delta.dot(self.layers[i].T)
        deltas.append(delta)
        deltas.reverse()
        return deltas  # last delta (partial)

    def iteration_by_deltas(self, inputs, deltas, alpha=0.01):
        for i in xrange(len(self.layers)):
            self.layers[i] -= alpha*inputs[i].T.dot(deltas[i+1])

    def derivative_by_next_net(self, inputs, delta):
        return self.derivative_by_deltas(inputs, self.deltas_by_next_net(inputs, delta))

    def derivative_by_deltas(self, inputs, deltas):
        derivatives = []
        for i in xrange(len(self.layers)):
            derivatives.append(inputs[i].T.dot(deltas[i+1]))
        return derivatives

    def iteration_by_derivative(self, derivatives, alpha=0.01):
        for i in xrange(len(self.layers)):
            self.layers[i] -= alpha*derivatives[i]

