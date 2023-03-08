import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt


def gauss(x, p):  # p[0]==mean, p[1]==stdev
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))


known_param = np.array([1545, 0.7])  # mean 2.0 sd 0.7
xmin, xmax = 1530, 1560
N = 1000
X = np.linspace(xmin, xmax, N)
Y = gauss(X, known_param)
# Add some noise
Y += .30*np.random.random(N)
plt.subplot(221)
plt.ylim(0, 0.9)
plt.plot(Y)

Yf = fft.ifft(Y)
plt.subplot(222)
plt.plot(Yf)

B = Yf

# Filtering noise
for a in range(len(B[10:990])):
    B[10+a] = 0

Bf = fft.fft(B)
plt.subplot(224)
plt.plot(B)

plt.subplot(223)
plt.ylim(0, 0.9)
plt.plot(Bf)
plt.show()
