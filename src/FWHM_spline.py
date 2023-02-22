# resource from https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
# adopted 22 February 2023
# modified with qfactor formula Q = peak lambda / delta half peak width value

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

def make_norm_dist(x, mean, sd):
    return 1.0/(sd*np.sqrt(2*np.pi))*np.exp(-(x - mean)**2/(2*sd**2))

#x = np.linspace(10, 110, 1000)
x = np.linspace(-1, 5, 1000)
green = make_norm_dist(x, 30, 3)
pink = make_norm_dist(x, 60, 4)

blue = green + pink
blue = green

# create a spline of x and blue-np.max(blue)/2 
spline = UnivariateSpline(x, blue-np.max(blue)/2, s=0)
r1, r2 = spline.roots() # find the roots

delta_hw = blue[int(r2)] - blue[int(r1)]
peak = np.max(blue)

qfactor = peak/delta_hw
print(delta_hw, peak, qfactor)

import pylab as pl
pl.plot(x, blue)
pl.axvspan(r1, r2, facecolor='g', alpha=0.5)
pl.show()