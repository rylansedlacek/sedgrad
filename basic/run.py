import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_engine import Value
from sedgrad.sed_nn import MLP
from sedgrad.sed_optim import Adam

# event_id: 0=100m,1=400m,2=1500m etc.
data = [
    ([0, 10.5, 10.0], 9.9),
    ([1, 50.2, 49.0], 48.5),
    ([2, 230.0, 225.0], 222.0),
]

def mse_loss(pred, target):
    diff = pred - target
    return diff * diff

def train(model, data, epochs=10000, lr=5):

    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for x_raw, y_raw in data:

            inputs = [Value(v) for v in x_raw]
            target = Value(y_raw)

            pred = model(inputs)
            loss = mse_loss(pred, target)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.data
        if epoch % 50 == 0:

            print(f"Epoch {epoch}, Loss: {total_loss / len(data)}")

def predict(model, event_id, pr, target):
    inputs = [Value(event_id), Value(pr), Value(target)]
    pred = model(inputs)
    return pred.data

def main():
    model = MLP(3, [10, 10, 1])  # small MLP

    train(model, data)

    while True:
        ev = int(input("Enter event ID (0=100m,1=400m,2=1500m): "))

        pr = float(input("Enter your PR time: "))

        target = float(input("Enter the time you want to hit: "))

        prediction = predict(model, ev, pr, target)
        
        print(f"Predicted next time you might run: {prediction:.2f} seconds")

if __name__ == "__main__":
    main()