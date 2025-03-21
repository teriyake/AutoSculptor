import numpy as np
from autosculptor.analysis.geodesic_calculator import CachedGeodesicCalculator
from autosculptor.core.data_structures import Sample, Stroke
from typing import List


class Utils:
	def full_stack():
		"""
		https://stackoverflow.com/a/16589622
		"""
		import traceback, sys

		exc = sys.exc_info()[0]
		stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
		if exc is not None:  # i.e. an exception is present
			del stack[-1]  # remove call of full_stack, the printed exception
			# will contain the caught exception caller instead
		trc = "Traceback (most recent call last):\n"
		stackstr = trc + "".join(traceback.format_list(stack))
		if exc is not None:
			stackstr += "  " + traceback.format_exc().lstrip(trc)
		return stackstr
