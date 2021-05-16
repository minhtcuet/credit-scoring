import numpy as np
import pandas as pd
import plotly.express as px

from credit_scoring.metrics.credit_score import CreditScore


class CalculatedKS(CreditScore):
	def ks(self) -> tuple:
		'''
		Calculate the KS table of each decile, KS value, the deicile at KS
		:return: KS Tabel, KS value, Decile
		'''

		# Create the dataframe contain target labels and probability values
		data = pd.DataFrame({"target": self.target, "prob": self.pred})

		# Calculate the probability of nonvents
		data['target0'] = 1 - data['target']

		# Find the suitable bucket
		self.bucket = min(self.bucket, data['prob'].nunique())

		# Cut the probability values to 'bucket' ranges by quantile
		data['bucket'] = pd.qcut(data['prob'], self.bucket)
		grouped = data.groupby('bucket', as_index=False)

		ks_table = pd.DataFrame()

		# Calculate the min, max of probability,
		ks_table['min_prob'] = grouped.min()['prob']
		ks_table['max_prob'] = grouped.max()['prob']

		# Calculate the number of events, nonevents
		ks_table['events'] = grouped.sum()['target']
		ks_table['nonevents'] = grouped.sum()['target0']

		ks_table = ks_table.sort_values(by="min_prob", ascending=False).reset_index(drop=True)
		ks_table['cum_eventrate'] = (ks_table.events / data['target'].sum()).cumsum()
		ks_table['cum_noneventrate'] = (ks_table.nonevents / data['target0'].sum()).cumsum()
		ks_table['KS'] = np.round(ks_table['cum_eventrate'] - ks_table['cum_noneventrate'], 3) * 100
		ks_table.loc[0, 'cum_eventrate'] = ks_table.loc[0, 'cum_noneventrate'] = 0
		ks_table.index = range(self.bucket)
		ks_table.index.rename('Decile', inplace=True)

		ks_val, decile = max(ks_table['KS']), ks_table.index[ks_table['KS'] == max(ks_table['KS'])][0]
		return ks_table, ks_val, decile

	def plot_ks_chart(self):
		data, ks_val, decile = self.ks()
		text = str(ks_val) + "%" + " at decile " + str(decile)
		fig = px.line(data, x=data.index, y=['cum_eventrate', 'cum_noneventrate'],
		              title='Kolmogorov-Smirnov statistic: ' + text)
		fig.add_vline(x=decile, line_width=3, line_dash="dash", line_color="green")
		fig.show()
