.. Assig2 documentation master file, created by
   sphinx-quickstart on Fri Oct 14 10:16:07 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Assig2's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Module: basic Python
	Assignment #4 (October 7, 2021)


	--- Goal
	Create a ProbabilityDensityFunction class that is capable of throwing
	preudo-random number with an arbitrary distribution.

	(In practice, start with something easy, like a triangular distribution---the
	initial debug will be easier if you know exactly what to expect.)


	--- Specifications
	- the signature of the constructor should be __init__(self, x, y), where
	  x and y are two numpy arrays sampling the pdf on a grid of values, that
	  you will use to build a spline
	- [optional] add more arguments to the constructor to control the creation
	  of the spline (e.g., its order)
	- the class should be able to evaluate itself on a generic point or array of
	  points
	- the class should be able to calculate the probability for the random
	  variable to be included in a generic interval
	- the class should be able to throw random numbers according to the distribution
	  that it represents
	- [optional] how many random numbers do you have to throw to hit the
	  numerical inaccuracy of your generator?
	


.. testcode::

	import numpy as np
	from scipy.interpolate import InterpolatedUnivariateSpline
	from matplotlib import pyplot as plt
	from scipy.optimize import curve_fit



	class ProbabilityDensityFunction(InterpolatedUnivariateSpline):

	    """Class describing a probability density function.
	    Parameters
	    ----------
	    x : array-like
		The array of x values to be passed to the pdf, assumed to be sorted.
	    y : array-like
		The array of y values to be passed to the pdf.
	    k : int
		The order of the splines to be created.
	    """

	    def __init__(self, x, y, k=3):
		"""Constructor.
		"""
		# Normalize the pdf, if it is not.
		norm = InterpolatedUnivariateSpline(x, y, k=k).integral(x[0], x[-1])
		y /= norm


		super().__init__(x, y, k=k)#super richiama i metodi di classi in altre classi. 
		#inheritance=la classe prende i metodi/ gli attributi da altre classi 
		# (da quella che è tra parentesi nel titolo della classe)


		ycdf = np.array([self.integral(x[0], xcdf) for xcdf in x]) 
		#Return definite integral of the spline between two given points.


		self.cdf = InterpolatedUnivariateSpline(x, ycdf, k=k)
		#1-D interpolating spline for a given set of data points.
		#Fits a spline y = spl(x) of degree k to the provided x, y data. 
		# Spline function passes through all provided points. 
		# Equivalent to UnivariateSpline with s=0.


		# Need to make sure that the vector I am passing to the ppf spline as
		# the x values has no duplicates---and need to filter the y
		# accordingly:
		xppf, ippf = np.unique(ycdf, return_index=True)
		yppf = x[ippf]
		self.ppf = InterpolatedUnivariateSpline(xppf, yppf, k=k)

	    def prob(self, x1, x2):
		"""Return the probability for the random variable to be included
		between x1 and x2.
		Parameters
		----------
		x1: float or array-like
		    The left bound for the integration.
		x2: float or array-like
		    The right bound for the integration.
		"""
		return self.cdf(x2) - self.cdf(x1)

	    def rnd(self, size=1000):
		"""Return an array of random values from the pdf.
		Parameters
		----------
		size: int
		    The number of random numbers to extract.
		"""
		return self.ppf(np.random.uniform(size=size))
	    


