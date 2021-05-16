import numpy as np
import pandas as pd
import plotly.graph_objects as go

from credit_scoring.metrics.credit_score import CreditScore


class CalculatedLift(CreditScore):
	def __init__(self, pred, target, bucket=10):
		super().__init__(pred, target)
		self.bucket = bucket
		self.pred = 1 - self.pred

	def lift(self) -> pd.DataFrame:
		"""

		:return: The lift table contains Gain, Lift each deciles
		"""

		# Create dataframe contain target and prediction values
		data = pd.DataFrame({'Target': self.target, 'Pred': self.pred})
		data = data.sort_values(by='Pred', ascending=False)

		# Cut Prediction values to q range by percentile
		data['decile'] = pd.qcut(data['Pred'], q=self.bucket)

		# Calculate number events and total records, PS: the bellow code only support pandas from version 1.0.0
		lift = data.groupby('decile').agg(
			Number_of_customers=('Target', 'count'),
			Number_of_goods=('Target', 'sum')
		)

		# Calculate cumulative good by sumup in each decile
		lift['Cumulative_goods'] = lift['Number_of_goods'].cumsum()

		# Calculate the distribution of Events in each decile
		lift['Percent_of_Events'] = lift['Number_of_goods'] / lift['Number_of_goods'].sum() * 100

		# Calculate Gain value by sumup of events distribution
		lift['Gain'] = lift['Percent_of_Events'].cumsum()

		# Calculate the random probability of events in each decile
		sample = [i * 100 / self.bucket for i in range(1, self.bucket + 1)]

		# The lift would be : Gain / random probability of each decile
		lift['Lift'] = lift['Gain'] / np.array(sample)
		return lift

	def plot_gain_chart(self) -> None:
		"""

		:return: Plot the Gain chart base on the result of lift function above
		"""
		data = self.lift()
		gain = data.Gain.tolist()
		gain.insert(0, 0)
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=list(np.linspace(0, 100, self.bucket + 1, dtype=int)),
		                         y=list(np.linspace(0, 100, self.bucket + 1, dtype=int)),
		                         mode='lines+markers',
		                         name='lines+markers'))
		fig.add_trace(go.Scatter(x=list(np.linspace(0, 100, self.bucket + 1, dtype=int)), y=gain,
		                         mode='lines+markers',
		                         name='lines+markers'))
		fig.update_xaxes(title_text="Percentage")
		fig.update_yaxes(title_text="% of Gain")
		fig.update_layout(title='Gain Charts')
		fig.show()

	def plot_lift_chart(self) -> None:
		"""

		:return: Plot the Lift chart base on the reuslt of lift function above
		"""
		data = self.lift()
		fig = go.Figure()
		fig.add_trace(go.Scatter(x=list(np.linspace(0, 100, self.bucket + 1, dtype=int)),
		                         y=np.repeat(1, self.bucket),
		                         mode='lines+markers',
		                         name='lines+markers'))
		fig.add_trace(go.Scatter(x=list(np.linspace(0, 100, self.bucket + 1, dtype=int)),
		                         y=data.Lift,
		                         mode='lines+markers',
		                         name='lines+markers'))
		fig.update_xaxes(title_text="Percentage")
		fig.update_yaxes(title_text="Lift")
		fig.update_layout(title='Lift Charts')
		fig.show()
