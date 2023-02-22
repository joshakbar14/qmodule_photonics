# resource from https://stackoverflow.com/questions/10582795/finding-the-full-width-half-maximum-of-a-peak
# adopted 22 February 2023
# modified with qfactor formula Q = peak lambda / delta half peak width value

import numpy as np
import scipy.optimize as opt

def gauss(x, p): # p[0]==mean, p[1]==stdev
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))

# Create some sample data
#known_param = np.array([2.0, .7]) #mean 2.0 sd 0.7
known_param = np.array([30.0, 3])
#xmin,xmax = -1.0, 5.0
xmin,xmax = 10, 110
N = 1000
X = np.linspace(xmin,xmax,N)
Y = gauss(X, known_param)

# Add some noise
Y += .10*np.random.random(N)

# Renormalize to a proper PDF
Y /= ((xmax-xmin)/N)*Y.sum()

# Fit a gaussian
p0 = [0,1] # Inital guess is a normal distribution
errfunc = lambda p, x, y: gauss(x, p) - y # Distance to the target function
p1, success = opt.leastsq(errfunc, p0[:], args=(X, Y))

fit_mu, fit_stdev = p1

FWHM = 2*np.sqrt(2*np.log(2))*fit_stdev
# print "FWHM", FWHM

print(p1)
print(FWHM)

## Plot data ##

from pylab import *
plot(X,Y)
plot(X, gauss(X,p1),lw=3,alpha=.5, color='r')
axvspan(fit_mu-FWHM/2, fit_mu+FWHM/2, facecolor='g', alpha=0.5)
show()