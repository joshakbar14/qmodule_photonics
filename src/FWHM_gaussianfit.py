# resource from https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
# adopted 22 February 2023
# modified with qfactor formula Q = peak lambda / delta half peak width value

import numpy as np
import scipy.optimize as opt
import scipy.fft as fft

def gauss(x, p): # p[0]==mean, p[1]==stdev
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))

## Sample Data ##
# Create some sample data
known_param = np.array([1545, 2]) #mean 2.0 sd 0.7
xmin,xmax = 1530, 1560
N = 1000
X = np.linspace(xmin,xmax,N)
Y = gauss(X, known_param)
# Add some noise
Y += .30*np.random.random(N)

# Renormalize to a proper PDF
# Y /= ((xmax-xmin)/N)*Y.sum()

# Fit a gaussian
p0 = [0,1555] # Inital guess is a normal distribution
errfunc = lambda p, x, y: gauss(x, p) - y # Distance to the target function
p1, success = opt.leastsq(errfunc, p0[:], args=(X, Y))

fit_mu, fit_stdev = p1

#FWHM for Gaussian
FWHM = 2*np.sqrt(2*np.log(2))*fit_stdev
# print "FWHM", FWHM

delta_hw = FWHM
qfactor = fit_mu/delta_hw

print(delta_hw, fit_mu, qfactor)
print("peak", fit_mu,
      "delta", delta_hw,
      "qfactor", qfactor)
print(p1)
print(FWHM/2)

## Plot data ##

from pylab import *
plot(X,Y)
xlim(1530,1560)
plot(X, gauss(X,p1)+max(Y)/4,lw=3,alpha=.5, color='r')
axvspan(fit_mu-FWHM/2, fit_mu+FWHM/2, facecolor='g', alpha=0.5)
show()