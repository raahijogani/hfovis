import numpy as np


class RingBuffer:
    def __init__(self, samples: int, channels: int):
        self.buf = np.empty((samples, channels))
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


class RebufferingView:
    def __init__(self, parent, buffer_size, overlap):
        self.parent = parent
        self.buffer_size = round(buffer_size)
        self.step = self.buffer_size - round(overlap)
        self.local_cursor = parent.cursor  # Absolute position for this view

    def read(self):
        needed_end = self.local_cursor + self.buffer_size
        if not self.parent._fill_up_to(needed_end):
            return None

        relative_start = self.local_cursor - self.parent.cursor
        relative_end = relative_start + self.buffer_size
        if relative_end > self.parent.buffer.shape[0]:
            return None

        window = self.parent.buffer[relative_start:relative_end]
        start_index = self.local_cursor
        self.local_cursor += self.step

        # truncate parent buffer to save memory
        self._trim_parent_buffer()

        timestamp = start_index / self.parent.sample_rate
        return window, start_index, timestamp

    def _trim_parent_buffer(self):
        min_cursor = min(v.local_cursor for v in self.parent.views)
        trim_amount = min_cursor - self.parent.cursor
        if trim_amount > 0:
            self.parent.buffer = self.parent.buffer[trim_amount:]
            self.parent.cursor += trim_amount


class MultiRebufferer:
    def __init__(self, streamer, sample_rate, pad_end=True):
        self.streamer = streamer
        self.sample_rate = sample_rate
        self.buffer = np.empty((0,))  # Will become (n_samples, n_channels)
        self.cursor = 0  # Absolute sample index of the start of buffer
        self.channels = None
        self.views = []
        self.pad_end = pad_end
        self.stream_ended = False

    def _initialize_buffer(self, data):
        if data.ndim == 1:
            data = data[:, np.newaxis]
        self.channels = data.shape[1]
        self.buffer = np.empty((0, self.channels))

    def _fill_up_to(self, needed_end):
        while self.cursor + self.buffer.shape[0] < needed_end and not self.stream_ended:
            new_data = self.streamer.read()
            if new_data is None or new_data.size == 0:
                self.stream_ended = True
                break
            if new_data.ndim == 1:
                new_data = new_data[:, np.newaxis]
            if self.channels is None:
                self._initialize_buffer(new_data)
            self.buffer = np.concatenate((self.buffer, new_data), axis=0)

        # Pad if needed
        if self.pad_end and self.stream_ended:
            current_end = self.cursor + self.buffer.shape[0]
            if current_end < needed_end:
                pad_amount = needed_end - current_end
                last_row = (
                    self.buffer[-1:]
                    if self.buffer.shape[0] > 0
                    else np.zeros((1, self.channels))
                )
                pad_block = np.repeat(last_row, pad_amount, axis=0)
                self.buffer = np.vstack([self.buffer, pad_block])

        return self.cursor + self.buffer.shape[0] >= needed_end

    def get_view(self, buffer_size, overlap):
        view = RebufferingView(self, buffer_size, overlap)
        self.views.append(view)
        return view
