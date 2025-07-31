import queue
import threading
from abc import ABC, abstractmethod

import numpy as np
from streamz import Stream


class Streamer(ABC):
    """
    Abstract base class for streaming data. Provides a common interface for
    starting, reading, and stopping a data stream.

    Subclasses must implement the `start`, `read`, and `stop` methods.
    """

    @abstractmethod
    def start(self):
        """Start the streaming process."""
        pass

    @abstractmethod
    def read(self, timeout=None):
        """
        Read the next chunk of data.

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for data before returning None. If None, blocks
            indefinitely.
        """
        pass

    @abstractmethod
    def stop(self):
        """Stop the streaming process."""
        pass


class DataStreamer(Streamer):
    """
    A simple data streamer that emits chunks of data at a specified rate.

    Parameters
    ----------
    data : np.ndarray
        The data to be streamed, should be a 2D array where each row is a
        chunk of data.
    chunk_size : int, default=1
        The size of each chunk to emit. Defaults to 1.
    interval_s : float, default=0.0
        The interval in seconds between emitted chunks. Defaults to 0.0 (no delay).
    max_queue_size : int, default=100
        Maximum size of the internal queue used to buffer emitted chunks.

    Attributes
    ----------
    data : np.ndarray
    chunk_size : int
    interval : float
    source : Stream
        The source stream that emits data chunks.
    throttled : Stream
        A throttled version of the source stream that limits the rate of emitted chunks.
    queue : queue.Queue
        A thread-safe queue for delivering chunks to the `read()` method.

    Methods
    -------
    start()
        Start the streaming process in a background thread.
    read(timeout=None)
        Retrieve the next chunk of data. Blocks by default; returns None when the stream
        ends.
    stop()
        Stop producing further data and wait for the producer thread to finish.
    """

    def __init__(
        self,
        data: np.ndarray,
        chunk_size: int = 1,
        interval_s: float = 0.0,
        max_queue_size: int = 100,
    ):
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

        Parameters
        ----------
        timeout : float, optional
            Maximum time to wait for data before returning None. If None, blocks
            indefinitely.
        """
        return self.queue.get(timeout=timeout)

    def stop(self):
        """Stop producing further data and wait for producer to finish."""
        self._stop_event.set()
        if self._producer.is_alive():
            self._producer.join()
