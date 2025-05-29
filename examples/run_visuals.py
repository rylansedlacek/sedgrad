import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_engine import Value
from sedgrad.sed_nn import MLP
from visualizer import draw_dot, visualize_network

a = Value(1.5)
b = Value(-2.0)
c = Value(3.0)
d = Value(-0.5)
e = Value(4.0)

x1 = a * b
x2 = c + d
x3 = x1 + x2
x4 = x3 * e
x5 = x4.tanh()

y1 = a + c
y2 = b * d
y3 = y1 * y2
y4 = y3.tanh()
y5 = y4 + e

z = x5 * y5

output = z.tanh()

output.backward()

dot = draw_dot(output)
dot.render('graph_large', view=True)

#--------------------

mlp = MLP(10, [20, 20, 10])
visualize_network(mlp, 10)