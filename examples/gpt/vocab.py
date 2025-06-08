# TOKENIZATION!

class Vocabulary:
    def __init__(self, text): # make everybody an int
        self.chars = sorted(list(set(text))) # all characters in text
        self.stoi = {ch: i for i, ch in enumerate(self.chars)} # char -> index 
        self.itos = {i: ch for ch, i in self.stoi.items()} # index -> string

    def encode(self, s): #convert the string into a list of ints which are the idices
        return [self.stoi[c] for c in s]

    def decode(self, tokens): # convert the list of indices back into strings
        return ''.join([self.itos[i] for i in tokens])
    