from sed_engine import Value
import random

class Neuron:
    def __init__(self, num_inputs):
        self.weights = []
        for i in range(num_inputs):
            weight = Value(random.uniform(-1,1))
            self.weights.append(weight)
        
        self.bias = Value(0.0)


    def __call__(self, inputs):
        total = self.bias

        for i in range(len(self.weights)):
            product = self.weights[i] * inputs[i]
            total += product

        output = total.tanh()
        return output


    def parameters(self):
        params = []

        for a in self.weights:
            params.append(a)
        params.append(self.bias)
        return params
        



        
  

