import random
import math
import sys
import os
from vocab import Vocabulary
from gpt import TinyLanguageModel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sedgrad.sed_engine import Value
from sedgrad.sed_nn import MLP

corpus = "the quick brown fox jumps over the lazy dog."

vocab = Vocabulary(corpus)
context_size = 4  # context window length

model = TinyLanguageModel(len(vocab.chars), context_size)

data = vocab.encode(corpus) # encode entire corpus into indices

learning_rate = 0.05
epochs = 10000

for epoch in range(epochs):
    ix = random.randint(0, len(data) - context_size - 1)    # pick random position in data for context and target
    context = data[ix:ix+context_size]
    target = data[ix+context_size]

    # forward pass
    pred = model.forward(context)

    probs = []
    for out in pred:
        if isinstance(out, list):
            probs.extend([p.data for p in out])
        else:
            probs.append(out.data)

    exps = [math.exp(p) for p in probs]     # compute softmax
    sum_exps = sum(exps)
    softmax = [e / sum_exps for e in exps]

    loss = -math.log(softmax[target])
    loss_val = loss  # no autograd here  just scalar for now

    loss_val = Value(loss) # backward pass
    loss_val.backward()

    # the confusing gradient descent step
    for param in model.parameters():
        param.data -= learning_rate * param.grad
        param.grad = 0.0

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, loss: {loss}")


# sample step
context = vocab.encode("qui")
generated = context.copy()

for _ in range(10):  # generate 10 characters
    out = model.forward([generated[-1]])

    if isinstance(out[0], list): 
         # flatten output to a list of probabilities
        flat_probs = [p.data for p in out[0]]
    else:
        flat_probs = [p.data for p in out]

    next_char_idx = max(range(len(flat_probs)), key=lambda i: flat_probs[i])     # pick the character with max probability
    generated.append(next_char_idx)

output = vocab.decode(generated)
print("generated:", output)
