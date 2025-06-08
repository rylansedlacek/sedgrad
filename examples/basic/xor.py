import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_nn import Neuron, Layer, MLP
from sedgrad.sed_engine import Value
from examples.visuals.visualizer import draw_dot, visualize_network

data = [
    ([0.0, 0.0], 0.0), # XOR = input1, input2, target value
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
]

mlp = MLP(2, [4, 1]) # 2 in - 4 hidden - 1 out

# the loop --------------------------
learning_rate = 0.1 
epochs = 100000

losses = []

for epoch in range(epochs): 
    total_loss = 0.0
    outputs = []

    for x, target in data:
        # (1)
        inputs = [Value(val) for val in x] # wrap the inputs in value objects
        target_val = Value(target)

        # (2)
        pred = mlp(inputs) # FORWARD PASS - prediction

        # (3)
        loss = (pred - target_val) * (pred - target_val) # compute loss (pred-target) ^ 2
        total_loss += loss.data
        outputs.append((x, pred.data))

        # (4)
        loss.backward()   # BACKWARD PASS - compute gradients

        # (5)
        for param in mlp.parameters():   # gradient descent: update parameters using gradients
            param.data -= learning_rate * param.grad

        # (6)
        for param in mlp.parameters():   # cear gradients for next example
            param.grad = 0.0

    avg_loss = total_loss / len(data)
    losses.append(avg_loss)

    if epoch % 10000 == 0 or epoch == epochs - 1:
        print(f"epoch {epoch}, loss: {avg_loss}, outputs: {outputs}\n")

    if epoch == 99999: # final run graph it
        dot = draw_dot(pred)
        dot.render('graph_large', view=True)
# the loop --------------------------