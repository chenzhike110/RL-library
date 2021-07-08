from collections import deque
import numpy as np

class Runningstate(object):
    def __init__(self, shape) -> None:
        super().__init__()
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    
    def push(self,x):
        x = np.asarray(x)
        assert x.shape == self._M.shape
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM) / self._n
            self._S[...] = self._S + (x - oldM) * (x - self._M)
    @property
    def n(self):
        return self._n

    @property
    def mean(self):
        return self._M

    @property
    def var(self):
        return self._S / (self._n - 1) if self._n > 1 else np.square(self._M)

    @property
    def std(self):
        return np.sqrt(self.var)

    @property
    def shape(self):
        return self._M.shape

class ZFilter:
    """
    y = (x-mean)/std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0) -> None:
        super().__init__()
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = Runningstate(shape)
    
    def __call__(self, x, update=True):
        if update:
            self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std + 1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return np.array(x,dtype=np.float32)
    
    def output_shape(self, input_shape):
        return input_shape.shape