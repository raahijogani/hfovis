# hfovis

A real-time GUI for detecting, denoising, and visualizing high-frequency oscillations.

## Installation

Clone this repository and install the dependencies listed in `pyproject.toml`.

It is recommended to use [uv](https://docs.astral.sh/uv/) to recreate the virtual
environment for this project. Run:

```bash
uv sync
```

This will create a virtual environment. Source or run the activation script in order to
activate the environment.

## Usage

![Screenshot](./hfovis.png)

### Functions

#### Events Panel
The Events panel allows for navigation through the pool of detected events. The left two
plots show the raw and filtered iEEG signal captured around the event. On the right is a
spectrogram and labels for the channel the selected event was detected on and whether it
was classified as a real of pseudo-HFO. There are four buttons for navigating through
the events:
- **First**: Jump to the first event in the pool.
- **Live**: Jump to the last event in the pool or keep up with the latest events as they
  are detected.
- **<** (previous): Jump to the previous event in the pool.
- **>** (next): Jump to the next event in the pool.

You can also use the spinbox to navigate through the events.

#### Denoising Panel

This panel shows a heatmap of the counts of real and pseudo-HFO events per channel.

#### Raster Plot

A raster plot of the detected events over channels and time. Green points indicate
events that have been classified as real HFOs, while red points indicate pseudo-HFOs.
White points indicate events that have not been classified yet. If "show pseudo events"
isn't checked, only real HFOs will be shown in green and unverified points in white. The
event shown in the Events panel is marked with an 'X' and centered in the plot.

You can change the length of the time window shown in the raster plot with the spin box.

#### Frequency Content Plot

This heatmap shows the counts of events per channel in each frequency bin from 0-600 Hz.
The frequency of each event is calculated by taking the peak of the Fourier transform of
the event.

### Demo

To run the demo in `main.py`, you can replace `demo_data.mat` with your own iEEG file to
simulate a live stream of data with the specified sampling rate.

### Custom Streams

If you want to work with your own stream, you will need to implement
`hfovis.data.streaming.Stream` with `start()`, `stop()`, and `read()` methods.

In general, once you have a `Stream` object, it can be fed into the 
`hfovis.detector.RealTimeDetector` as follows:

```python
stream = MyStream()  # Your custom stream implementation

detector = RealTimeDetector(
    stream=stream,
    handle=main.handle,  # The handle to the GUI. (see main.py)
    fs=fs, # Sampling rate of the stream
    # ...
    # Check the docstring of `hfovis.detector.RealTimeDetector._default_config` for 
    # more parameters
)
```
