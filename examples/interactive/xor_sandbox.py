import streamlit as st
import math
import random
from dataclasses import dataclass

# to run: streamlit run examples/interactive/xor_sandbox.py

# sedgrad/sed_engine.py
class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def tanh(self):
        x = self.data
        t = math.tanh(x)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def __hash__(self):
        return id(self)

# sedgrad/sed_nn.py
class Neuron:
    def __init__(self, num_inputs):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(0.0)

    def __call__(self, inputs):
        total = self.bias
        for w, x in zip(self.weights, inputs):
            total += w * x
        return total.tanh()

    def parameters(self):
        return self.weights + [self.bias]

class Layer:
    def __init__(self, num_inputs, num_outputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, inputs):
        outputs = [n(inputs) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, num_inputs, layer_sizes):
        sizes = [num_inputs] + layer_sizes
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(layer_sizes))]

    def __call__(self, inputs):
        out = inputs
        for layer in self.layers:
            out = layer(out)
        return out

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

# sedgrad/sed_optim.py
class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {p: 0.0 for p in parameters}
        self.v = {p: 0.0 for p in parameters}
        self.t = 0

    def step(self):
        self.t += 1
        for p in self.parameters:
            g = p.grad
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g ** 2)
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

# examples/basic/xor.py

st.title("XOR Sandbox") # stream lit setup
lr = st.slider(" Choose Learning Rate", 0.001, 0.1, 0.01)
epochs = st.slider(" Choose Epochs", 1000, 20000, 5000, step=1000)
hidden = st.slider(" Choose Hidden Layer Size", 1, 10, 4)

if st.button("Train the model!"): # when they want to train
    data = [ 
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ]

    mlp = MLP(2, [hidden, 1])
    optimizer = Adam(mlp.parameters(), lr=lr)

    # the loop --------------------------
    for _ in range(epochs):
        total_loss = 0.0
        for x, y in data:
            inputs = [Value(i) for i in x]
            target = Value(y)
            pred = mlp(inputs)
            loss = (pred - target) * (pred - target)
            total_loss += loss.data
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    st.success(f"Training success! Avg Loss: {total_loss / len(data):.4f}") # more streamlit

    for x, y in data:
        pred = mlp([Value(i) for i in x])
        st.write(f"Input: {x} -> Predicted: {pred.data:.4f} | Target: {y}") # and finally
