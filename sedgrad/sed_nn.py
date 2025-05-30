from sedgrad.sed_engine import Value
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
        
class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [] # create a list of neurons
        for i in range(num_outputs):
             neuron = Neuron(num_inputs)
             self.neurons.append(neuron)

    def __call__(self, inputs):
        outputs = [] # pass the inputs to each neuron in the layer
        for neuron in self.neurons:
            output = neuron(inputs)
            outputs.append(output)

        if len(outputs) == 1: # if only one return the single val instead of list
            return outputs[0]
        else:
            return outputs
        
    def parameters(self):
        params = [] # collect params 
        for neuron in self.neurons:
            neuron_params = neuron.parameters()
            for p in neuron_params:
                params.append(p)
        return params

class MLP:
    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes
        self.layers = [] # create a list layer

        for i in range(len(layer_sizes)):
            input_size = sizes[i]
            output_size = sizes[i + 1]
            layer = Layer(input_size, output_size)
            self.layers.append(layer)
    
    def __call__(self, inputs):
        current = inputs  # pass inputs through each layer in order
        for layer in self.layers:
            current = layer(current)
        return current

    def parameters(self):
        params = [] #  collect all parameters from all layers
        for layer in self.layers:
            layer_params = layer.parameters()
            for p in layer_params:
                params.append(p)
        return params



        
  

