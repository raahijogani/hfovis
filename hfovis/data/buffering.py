import numpy as np


class RingBuffer:
    """
    A simple ring buffer implementation for storing samples in a circular manner.

    Parameters
    ----------
    samples : int
        The total number of samples the buffer can hold.
    channels : int
        The number of channels in the data.

    Attributes
    ----------
    buf : np.ndarray
        The underlying buffer storing the samples.
    n : int
        The total number of samples the buffer can hold.
    wptr : int
        The write pointer indicating the next position to write in the buffer.

    Methods
    -------
    write(chunk: np.ndarray)
        Writes a chunk of data into the buffer, wrapping around if necessary.
    read(start_idx: int, length: int) -> np.ndarray
    """

    def __init__(self, samples: int, channels: int):
        self.buf = np.empty((samples, channels))
        self.n = samples
        self.wptr = 0

    def write(self, chunk):
        """
        Writes a chunk of data into the buffer, wrapping around if necessary.

        Parameters
        ----------
        chunk : np.ndarray
            The chunk of data to write into the buffer. It should be a 2D array
            with shape (n_samples, n_channels).
        """
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
        """
        Reads a chunk of data from the buffer starting at `start_idx` for `length`
        samples. The read operation wraps around the buffer if necessary.

        Parameters
        ----------
        start_idx : int
            The starting index from which to read in the buffer.
        length : int
            The number of samples to read from the buffer.

        Returns
        -------
        np.ndarray
            The chunk of data read from the buffer. It will be a 2D array with
            shape (length, n_channels).
        """
        off = start_idx % self.n
        end = off + length
        if end <= self.n:
            return self.buf[off:end]
        first = self.n - off
        return np.concatenate((self.buf[off:], self.buf[: length - first]), 0)
