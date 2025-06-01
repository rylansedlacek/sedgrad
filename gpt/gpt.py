import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sedgrad.sed_engine import Value
from sedgrad.sed_nn import MLP

class TinyLanguageModel:
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size  # store the num of characters
        self.mlp = MLP(vocab_size, [hidden_size, vocab_size]) # build an MLP
                                                              # input = vocab_size
                                                              # hidden layers = hidden + output layers

    def forward(self, indices):
        inputs = [] # convert the list of indices into one-hot vectors
        for idx in indices:
            one_hot = [0.0] * self.vocab_size
            one_hot[idx] = 1.0
            input_vals = [Value(x) for x in one_hot] # wrap each element in the one-hot vector as a Value for autograd
            inputs.append(input_vals)

       # pass each one-hot encoded token through the MLP to get output logits
        outputs = [self.mlp(input_vector) for input_vector in inputs]  
        return outputs

    def parameters(self):
        return self.mlp.parameters()
    
   

  
