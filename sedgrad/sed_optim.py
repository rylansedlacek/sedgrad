class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):

        self.parameters = parameters
        self.lr = lr # the learning rate
        self.beta1 = beta1 # decay for first moment 
        self.beta2 = beta2 # dacay for second moment
        self.eps = eps 

        self.m = {p: 0.0 for p in self.parameters}  
        self.v = {p: 0.0 for p in self.parameters}  
        self.t = 0  # timestep

    def step(self):

        self.t += 1 
        for p in self.parameters:
            g = p.grad  

            # update biased first moment estimate
            self.m[p] = self.beta1 * self.m[p] + (1 - self.beta1) * g

            # Uddate biased second raw moment estimate
            self.v[p] = self.beta2 * self.v[p] + (1 - self.beta2) * (g ** 2)

            # compute bias corrected estimates based on w3 schools form
            m_hat = self.m[p] / (1 - self.beta1 ** self.t)
            v_hat = self.v[p] / (1 - self.beta2 ** self.t)

            p.data -= self.lr * m_hat / (v_hat ** 0.5 + self.eps)

    def zero_grad(self):
        for p in self.parameters: # a reset if you will 
            p.grad = 0.0
