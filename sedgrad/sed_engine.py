import math

class Value:
    # node
    def __init__(self, data, _children=(), _op=''):
        self.data = data 
        self.grad = 0.0 

        self._backward = lambda: None 
        self._prev = set(_children) 
        self._op = _op 

    def __repr__(self):
        return f"Value (data={self.data}, grad={self.grad})" 
    
    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)  # convert to Value if not already

        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out
    
    def __hash__(self): # a hashing addition to be safe
        return id(self)
    
    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)  # convert to Value if not already

        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out
    
    def __neg__(self):  # allows -a for Value a
        return self * -1

    def __sub__(self, other):  # allows a - b
        return self + (-other)
    
    # activation
    def tanh(self):
        x = self.data
        t = math.tanh(x)

        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad  
        out._backward = _backward

        return out
  
    # backpropagation
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
  