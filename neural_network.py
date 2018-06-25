import numpy as np

def last(li):
    return li[len(li)-1]

class neural_network():
    def __init__(self,layersNum,functions):
        self.layers = []
        self.biases = []
        assert len(layersNum)-1 == len(functions)
        self.nonlin_funcs = functions
        layers = list(layersNum)
        for i in range(len(layers)-1):
            self.layers.append(np.random.rand(layers[i],layers[i+1])-0.5)
            self.biases.append(np.random.rand(layers[i+1])-0.5)

    def hypot(self,input):
        for i in range(len(self.layers)):
            input = self.nonlin_funcs[i](input.dot(self.layers[i])+self.biases[i])
        return input

    def cost(self,input,label):
        return np.sum((self.hypot(input)-label)**2)

    def iteration(self,input,label,alpha=0.01):
        inputs = []
        deltas = []

        for i in range(len(self.layers)):
            inputs.append(input)
            input = self.nonlin_funcs[i](input.dot(self.layers[i])+self.biases[i])

        error = label-input
        deltas.append(error*last(self.nonlin_funcs)(input,True))

        for i in range(len(inputs)-1,0,-1):
            tmpdelta = last(deltas).dot(self.layers[i].T)*self.nonlin_funcs[i](inputs[i],True)
            deltas.append(tmpdelta)

        deltas.reverse()
        inputs.append(input)

        for i in range(len(self.layers)):
            self.layers[i] = self.layers[i] + alpha*inputs[i].T.dot(deltas[i])/len(inputs[0])
            t_del = deltas[i].T
            for j in range(len(self.biases[i])):
                self.biases[i][j] += alpha*sum(t_del[j])