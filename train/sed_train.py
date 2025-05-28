import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_nn import Neuron, Layer
from sedgrad.sed_engine import Value

if __name__ == "__main__":
    n = Neuron(3)
    x = [Value(1.0), Value(2.0), Value(3.0)]
    out = n(x)
    print("neuron output:", out.data)

    layer = Layer(3, 2)
    out = layer(x)
    print("layer output:", [o.data for o in out])