import numpy as np
import scipy.fft as fft
import matplotlib.pyplot as plt
from genAlg import LorentzianCurveFit


def gauss(x, p):  # p[0]==mean, p[1]==stdev
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))


known_param = np.array([1545, 0.7])  # mean 2.0 sd 0.7
xmin, xmax = 1530, 1560
N = 1000
X = np.linspace(xmin, xmax, N)
Y = gauss(X, known_param)
# Add some noise
Y += .30*np.random.random(N)

fitting = LorentzianCurveFit(X,Y)
fitting.fit()
fitting.plot()
