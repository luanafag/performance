# intern function to be used with RandomizedSearch

class RBGUniform:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, rng):
        return float(rng.uniform(self.a,self.b))
    
class RBGRandInt:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self, rng):
        return int(rng.integers(self.a,self.b+1))