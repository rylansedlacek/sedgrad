import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data # scalar value
        self.grad = 0.0 # gradient of the value

        self._backward = lambda: None # backpropagation function
        self._prev = set(_children) #input vals to compute 
        self._op = _op # debug


    def __repr__(self):
        return f"Value (data={self.data}, grad={self.grad})" 
    

    def __add__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            Value(other) # convert to Value if not already

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __mul__(self, other):
        if isinstance(other, Value):
            other = other
        else:
            Value(other) # convert to Value if not already

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    # activation
    def tanh(self):
        x = self.data
        t = math.tanh(x)

        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad  
        out._backward = _backward

        return out
  
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0  

        for node in reversed(topo):
            node._backward()


x = Value(2.0)   # input
w = Value(-3.0)  # weight
b = Value(1.0)   # bias

y = w * x + b    # weighted sum
o = y.tanh()     # activation

o.backward()     # compute gradients

print("Output:", o)
print("Gradients:")
print("  x.grad =", x.grad)
print("  w.grad =", w.grad)
print("  b.grad =", b.grad)
  