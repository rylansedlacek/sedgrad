from vocab import Vocabulary
from gpt import TinyLanguageModel
import random

corpus = "the quick brown fox jumps over the lazy dog."  # this creates out vocab stuff!
vocab = Vocabulary(corpus) # send it on over
context_size = 4

model = TinyLanguageModel(len(vocab.chars), context_size) # create our model 

# genesis begins
context = [0] * context_size  # start with all zeros 
output = []

for _ in range(100): # forward pass
    pred = model.forward(context) # softmax
    probs = [math.exp(p.data) for p in pred]
    total = sum(probs)
    probs = [p / total for p in probs]

    idx = random.choices(range(len(probs)), weights=probs)[0] # sample the next character with weights
    output.append(idx)
 
    context = context[1:] + [idx] # update our context windo

print(vocab.decode(output))  # and decode so we can see it

