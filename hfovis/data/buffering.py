import numpy as np


class RingBuffer:
    def __init__(self, samples: int, channels: int):
        self.buf = np.zeros((samples, channels))
        self.n = samples
        self.wptr = 0

    def write(self, chunk):
        n = len(chunk)
        end = self.wptr + n
        if end <= self.n:
            self.buf[self.wptr : end] = chunk
        else:
            first = self.n - self.wptr
            self.buf[self.wptr :] = chunk[:first]
            self.buf[: n - first] = chunk[first:]
        self.wptr = (self.wptr + n) % self.n

    def read(self, start_idx, length):
        off = start_idx % self.n
        end = off + length
        if end <= self.n:
            return self.buf[off:end]
        first = self.n - off
        return np.concatenate((self.buf[off:], self.buf[: length - first]), 0)
