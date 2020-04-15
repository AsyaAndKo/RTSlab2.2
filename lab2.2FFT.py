import numpy as np
import random
from math import sin
import matplotlib.pyplot as plt


# expected value
def expectation(x, N):
    return np.sum(x)/N


# standard deviation
def dispersion(x, mx, N):
    return np.sum((x - mx)**2)/N


def frequency(n, w):
    return w - n * number


def DFT(signal):
    n = len(signal)
    f = np.zeros(n, dtype=complex)
    for p in range(n):
        for k in range(n):
           f[p] += complex(signal[k]*np.cos((2*np.pi*p*k)/n), -signal[k]*np.sin((2*np.pi*p*k)/n))
    return f


def FFT(signal):
    n = len(signal)
    f = np.zeros(n, dtype=complex)
    if n % 2 == 0:
        even = FFT(signal[0::2])
        odd = FFT(signal[1::2])
        W = lambda k: np.exp(-2j * np.pi * k / n)
        for k in range(n // 2):
            f[k] = even[k] + W(k) * odd[k]
            f[k + n // 2] = even[k] - W(k) * odd[k]
    elif n == 1:
        return signal
    else:
        raise ValueError("Signal list's size has to be the power of 2")
    return f


# values for 11 variant
n = 10
w = 1500
N = 256
number = w/(n - 1)

# frequency
w_values = [frequency(n, w) for n in range(n)]
harmonics = np.zeros((n, N))
resulted = np.array([])

# generating harmonics
for n in range(n):
    amplitude = random.choice([i for i in range(-10, 10) if i != 0])
    phi = random.randint(-360, 360)
    for t in range(N):
        harmonics[n, t] = amplitude * sin(w_values[n] * t + phi)

# last harmony
for i in harmonics.T:
    resulted = np.append(resulted, np.sum(i))

# plot for random harmony (x - time, y - signal)
plt.figure(figsize=(50, 10))
plt.plot(resulted)
plt.grid(True)
plt.show()

fft = FFT(resulted)
# plot for amplitude spectrum (x - sinusoid, y - amplitude)
plt.figure(figsize=(50, 10))
plt.plot([np.sqrt(z.real*z.real + z.imag*z.imag)/len(fft) for z in fft])
plt.grid(True)
plt.show()

# plot for phase spectrum (x - sinusoid, y - phase)
plt.figure(figsize=(50, 10))
plt.plot([np.arctan(z.imag/z.real) for z in fft])
plt.grid(True)
plt.show()

