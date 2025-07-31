import threading
import queue
import numpy as np
from streamz import Stream
from abc import ABC, abstractmethod


class Streamer(ABC):
    """
    Abstract base class for data streamers.
    Subclasses must implement the start, read, and stop methods.
    """

    @abstractmethod
    def start(self):
        """Start the streaming process."""
        pass

    @abstractmethod
    def read(self, timeout=None):
        """
        Read the next chunk of data.
        If timeout is specified, waits at most that many seconds.
        Returns None when the stream ends.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the streaming process."""
        pass


class DataStreamer(Streamer):
    def __init__(
        self,
        data: np.ndarray,
        chunk_size: int = 1,
        interval_s: float = 0.0,
        max_queue_size: int = 100,
    ):
        """
        data           : NumPy array of shape (num_samples, num_channels)
        chunk_size     : number of samples per chunk
        interval_s     : interval between sending chunks in seconds
        max_queue_size : max number of chunks to buffer
        """
        self.data = data
        self.chunk_size = chunk_size
        # Rate limit interval in seconds
        self.interval = interval_s

        # Create a stream that starts its own Tornado IOLoop in a separate thread
        self.source = Stream(asynchronous=False)

        # Throttle events to at most one per `self.interval` seconds
        self.throttled = self.source.rate_limit(self.interval)

        # Thread-safe queue for delivering chunks to read()
        self.queue = queue.Queue(maxsize=max_queue_size)

        # Sink throttled stream into the queue
        self.throttled.sink(self.queue.put)

        # Internal thread to push raw data into stream
        self._stop_event = threading.Event()
        self._producer = threading.Thread(target=self._produce, daemon=True)

    def _produce(self):
        """Emit all chunks rapidly; rate_limit will space them out."""
        for start in range(0, len(self.data), self.chunk_size):
            if self._stop_event.is_set():
                break
            chunk = self.data[start : start + self.chunk_size]
            self.source.emit(chunk)

        # Signal end-of-stream
        self.source.emit(None)

    def start(self):
        """Begin streaming in background."""
        if not self._producer.is_alive():
            self._producer.start()

    def read(self, timeout=None):
        """
        Retrieve next chunk.
        Blocks by default; returns None when stream ends.
        """
        return self.queue.get(timeout=timeout)

    def stop(self):
        """Stop producing further data and wait for producer to finish."""
        self._stop_event.set()
        if self._producer.is_alive():
            self._producer.join()
