import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_engine import Value
from sedgrad.sed_nn import MLP
from sedgrad.sed_optim import Adam

model = MLP(2, [4, 4, 1])
optimizer = Adam(model.parameters(), lr=0.001)

for epoch in range(500):
    x = [Value(1.0), Value(2.0)]
    y_true = Value(1.0)

    y_pred = model(x) # forward pass 
    loss = (y_pred - y_true) * (y_pred - y_true)  # MSE

    optimizer.zero_grad() # backward pass
    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.data}") # each step ew
