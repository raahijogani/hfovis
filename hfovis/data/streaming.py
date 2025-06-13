import time
import numpy as np
from multiprocessing import Process, Queue


class DataStreamer:
    def __init__(
        self,
        data: np.ndarray,
        chunk_size: int = 1,
        fs: float = 0.0,
        max_queue_size: int = 100,
    ):
        """
        data           : NumPy array of shape (num_samples, num_channels)
        chunk_size     : number of samples per send()
        fs             : sampling frequency in Hz, used to determine the interval between chunk_size
        max_queue_size : max number of chunks to buffer
        """
        self.data = data
        self.chunk_size = chunk_size
        self.interval = 1 / fs if fs > 0 else 0.0
        self.queue = Queue(maxsize=max_queue_size)
        self._process = Process(target=self._stream)
        self._running = False

    def _stream(self):
        """Send chunks into the queue."""
        try:
            for start in range(0, len(self.data), self.chunk_size):
                end = start + self.chunk_size
                chunk = self.data[start:end]
                self.queue.put(chunk)  # blocks if queue is full
                if self.interval > 0:
                    time.sleep(self.interval)
        finally:
            self.queue.put(None)  # signal end of stream

    def start(self):
        if not self._running:
            self._process.start()
            self._running = True

    def read(self, timeout=None):
        """
        Get next chunk. Blocks by default.
        If timeout is specified, waits at most that many seconds.
        Returns None when stream ends.
        """
        return self.queue.get(timeout=timeout)

    def stop(self):
        if self._running:
            self._process.join()
            self._process.terminate()
            self._running = False

    def get_queue(self):
        """Access the queue for external use (e.g. in other processes)."""
        return self.queue
