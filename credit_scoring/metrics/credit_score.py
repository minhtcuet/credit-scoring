import numpy as np
import pandas as pd


class CreditScore:
	"""
	This class will calculated the KS value. (Only support binary classification - Credit Scoring for example)
	KS is defined as maximal absolute difference between CDFs of scores of good and bad clients

	PS: This version only support for continuous variable
	Random Variable will be able later version.
	"""

	def __init__(self, pred, target, bucket=200):
		"""

		:param pred: The value of prediction
		:param target: The target relevant
		"""
		self.pred = pred
		self.target = target
		self.check_valid_type()
		self.bucket = min(bucket, len(self.pred), len(self.target))

	def check_valid_type(self) -> bool:
		"""
		Check valid input type. Only supported np.array, pd.Seris, list
		:return:
		"""
		# Check supported type input
		if type(self.pred) not in [list, np.ndarray, pd.core.series.Series]:
			error_msg = "Your input type must be one of following: <class 'pandas.core.series.Series'>, " \
			            "<class 'numpy.ndarray'>, <class 'list'>"
			raise ValueError(error_msg)

		# Compare length
		if len(self.pred) != len(self.target):
			error_msg = "All arguments should have the same length."
			raise ValueError(error_msg)

		# Convert to numpy array and check the shape
		try:
			self.pred = np.array(self.pred, dtype=float)
			self.target = np.array(self.target, dtype=float)
			if len(self.pred.shape) != 1 or len(self.target.shape) != 1:
				error_msg = "Your inputs shape must be one demension"
				raise ValueError(error_msg)
		except Exception as e:
			raise ValueError(e)
		return True
