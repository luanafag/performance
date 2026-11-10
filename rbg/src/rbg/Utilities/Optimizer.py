import numpy as np
import numpy.typing as npt
from collections.abc import Callable 

class AdamOptimizer():
    def __init__(self, 
                 learning_rate: float = 0.001, 
                 tolerance: float = 1e-9, 
                 max_iter: int = 1000, 
                 delta: float = 1e-5,
                 beta1: float = 0.9,
                 beta2: float = 0.999, 
                 eps: float = 1e-8
                 ):
        self.lr = learning_rate
        self.tol = tolerance
        self.max_iter = max_iter
        self.delta = delta 
        self.beta1 = beta1
        self.beta2 = beta2 
        self.eps = eps 

    def _num_grad(self, 
               f: Callable[[np.ndarray], float], 
               x: np.ndarray
        ) -> np.ndarray:
        grad = np.zeros_like(x)

        for i in range(x.size):
            step = np.zeros_like(x)
            step[i] = self.delta
            grad[i] = (f(x + step) - f(x - step)) / (2 * self.delta)

        return grad 

    def getmin(self, 
               teta: npt.ArrayLike,
               f: Callable[[np.ndarray], float], 
               g: Callable[[np.ndarray], np.ndarray] | None = None
        ) -> tuple[bool, np.ndarray]:
        """
        Implementation of the Adam Optimizer, as described in the original paper
        
        Args:
            teta: Starting posion
            f: The callable function to optimize
            g: An optional callable gradient 
        
        Returns:
            A nparray with the point that minimizes f
        """

        x = np.atleast_1d(np.array(teta, dtype=float))

        grad = g if g is not None else (lambda x : self._num_grad(f, x))

        m: np.ndarray = np.zeros_like(x) 
        v: np.ndarray = np.zeros_like(x)
        converged = False
        
        for t in range(1, self.max_iter+1):
            
            g_teta = grad(x)
            m = self.beta1*m + (1-self.beta1)*g_teta
            v = self.beta2*v + (1-self.beta2)*g_teta**2
            M = m/(1-self.beta1**t)
            V = v/(1-self.beta2**t)
            n_x = x - self.lr*M/(V**0.5 + self.eps)

            if np.linalg.norm(n_x-x) < self.tol:
                x = n_x 
                converged = True
                break

            x = n_x 
            
        return converged, x

        
    