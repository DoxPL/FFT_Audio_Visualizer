import pyaudio
import matplotlib.pyplot as plt
from numpy import linspace, abs
from cmath import exp, pi

# Constants
DATA_CHUNK = 256
CHUNK_NUM = 100000
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Discrete Fourier Transform O(N^2)
def dft(audio_samples):
    res = []
    sample_count = len(audio_samples)
    for k in range(sample_count):
        tmp = 0.j
        for i in range(sample_count):
            exponential_num = -2j * pi * i * k / sample_count
            tmp += audio_samples[i] * exp(exponential_num)
        res.append(tmp)
    return res

# Fast Fourier Transform O(N*logN)
def fft(x):
    N = len(x)
    if N == 1:
        return [x[0]]
    else:
        fft_even = fft(x[0::2])
        fft_odd = fft(x[1::2])

        for k in range(0, N // 2):
            p = fft_even[k]
            q = exp(-2j * pi / N * k) * fft_odd[k]
            fft_even[k] = p + q
            fft_odd[k] = p - q

        return fft_even + fft_odd

# Frequency plot for a given chunk
def freq_plot(data_x, data_y, ax):
    ax.plot(data_x, 2.0 / DATA_CHUNK * abs(data_y[:DATA_CHUNK//2]))
    plt.pause(0.002)
    ax.clear()
    plt.grid()
    ax.set_title('FFT')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Magnitude')


if __name__ == '__main__':
    pyaudio = pyaudio.PyAudio()

    stream = pyaudio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=DATA_CHUNK)

    T = 1 / RATE
    data_x = linspace(0.0, 1.0 / (2.0 * T), int(DATA_CHUNK / 2))
    plt.ion()
    _, ax = plt.subplots()
    plt.draw()

    for _ in range(0, CHUNK_NUM):
        data = stream.read(DATA_CHUNK, exception_on_overflow=False)
        data_fft = fft(data)
        freq_plot(data_x, data_fft, ax)
