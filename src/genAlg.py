import numpy, scipy, matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

class LorentzianCurveFit:
    def __init__(self, xData, yData):
        self.xData = xData
        self.yData = yData

    # def func(self, x, a, b, offset): #exponential curve fitting function
    #     return a * numpy.exp(-b*x) + offset
    
    def func(self, x, a, b, c, offset): # Lorentzian curve fitting function
        return offset + (a / (1.0 + ((x - b) / c) ** 2.0))

    # function for genetic algorithm to minimize (sum of squared error)
    def sumOfSquaredError(self, parameterTuple):
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = self.func(self.xData, *parameterTuple)
        return numpy.sum((self.yData - val) ** 2.0)

    def generate_Initial_Parameters(self):
        # min and max used for bounds
        maxX = max(self.xData)
        minX = min(self.xData)
        maxY = max(self.yData)
        minY = min(self.yData)

        parameterBounds = []
        # parameterBounds.append([-0.185, maxX]) # search bounds for a
        # parameterBounds.append([minX, maxX]) # search bounds for b
        # parameterBounds.append([0.0, maxY]) # search bounds for Offset

        # Lorentzian Function Fit
        parameterBounds.append([0.0, maxY]) # search bounds for a
        parameterBounds.append([minX, maxX]) # search bounds for b
        parameterBounds.append([0.0, maxX-minX]) # search bounds for c
        parameterBounds.append([minY, maxY]) # search bounds for Offset

        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(self.sumOfSquaredError, parameterBounds, seed=3)
        return result.x

    def fit(self):
        # by default, differential_evolution completes by calling
        # curve_fit() using parameter bounds
        geneticParameters = self.generate_Initial_Parameters()
        print('fit with parameter bounds (note the -0.185)')
        print(geneticParameters)
        print()

        # second call to curve_fit made with no bounds for comparison
        fittedParameters, pcov = curve_fit(self.func, self.xData, self.yData, geneticParameters)

        print('re-fit with no parameter bounds')
        print(fittedParameters)
        print()

        modelPredictions = self.func(self.xData, *fittedParameters) 

        absError = modelPredictions - self.yData

        SE = numpy.square(absError) # squared errors
        MSE = numpy.mean(SE) # mean squared errors
        RMSE = numpy.sqrt(MSE) # Root Mean Squared Error, RMSE
        Rsquared = 1.0 - (numpy.var(absError) / numpy.var(self.yData))

        print()
        print('RMSE:', RMSE)
        print('R-squared:', Rsquared)
        print()

        self.fittedParameters = fittedParameters
        self.RMSE = RMSE
        self.Rsquared = Rsquared

    def plot(self, graphWidth=800, graphHeight=600):
        f = plt.figure(figsize=(graphWidth/100.0, graphHeight/100.0), dpi=100)
        axes = f.add_subplot(111)

        # first the raw data as a scatter plot
        axes.plot(self.xData, self.yData,  'D')

        # create data for the fitted equation plot
        xModel = numpy.linspace(min(self.xData), max(self.xData))
        yModel = self.func(xModel, *self.fittedParameters)

        # now the model as a line plot
        axes.plot(xModel, yModel)

        axes.set_xlabel('X Data') # X axis data label
        axes.set_ylabel('Y Data') # Y axis
    
        plt.show()
        plt.close('all')  # clean up after using pyplot
