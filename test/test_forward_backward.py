import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_nn import Layer
from sedgrad.sed_engine import Value

x = [Value(1.0), Value(2.0), Value(-1.0)]

layer = Layer(3, 1)

out = layer(x)
print("Output value:", out.data)

out.backward()

for i, param in enumerate(layer.parameters()):
    print(f"Param {i} value: {param.data:.4f}, grad: {param.grad:.4f}")
