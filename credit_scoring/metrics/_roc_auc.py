import pandas as pd
import plotly.express as px
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

from credit_scoring.metrics.credit_score import CreditScore


class CalculatedAUC(CreditScore):

	def gini(self) -> float:
		"""
		Calculated the gini value base on prediction values and target values
		:return: GINI value
		"""
		return 2 * roc_auc_score(self.target, self.pred) - 1

	def roc_auc(self) -> float:
		"""
		Calculated the roc_auc value base on prediction values and target values
		:return: The ROC_AUC value
		"""
		return roc_auc_score(self.target, self.pred)

	def plot_roc_curve(self) -> None:
		"""
		Using plotly to plot roc curve
		:return: None
		"""
		fpr, tpr, thresholds = roc_curve(self.target, self.pred)
		fig = px.area(
			x=fpr, y=tpr,
			title=f'ROC Curve (AUC={auc(fpr, tpr):.4f})',
			labels=dict(x='False Positive Rate', y='True Positive Rate'),
			width=700, height=500
		)
		fig.add_shape(
			type='line', line=dict(dash='dash'),
			x0=0, x1=1, y0=0, y1=1
		)

		fig.update_yaxes(scaleanchor="x", scaleratio=1)
		fig.update_xaxes(constrain='domain')
		fig.show()

	def plot_compare_true_label(self) -> None:
		"""
		Using plotly to display how the true labels better than false labels
		:return: None
		"""
		fig_hist = px.histogram(
			x=self.pred, color=self.target, nbins=50,
			labels=dict(color='True Labels', x='Score')
		)
		fig_hist.show()

	def plot_tpr_fpr(self) -> None:
		"""
		Using plotly to plot the relation between TPR and FPR at every threshold
		:return: None
		"""
		fpr, tpr, thresholds = roc_curve(self.target, self.pred)
		df = pd.DataFrame({
			'False Positive Rate': fpr,
			'True Positive Rate': tpr
		}, index=thresholds)
		df.index.name = "Thresholds"
		df.columns.name = "Rate"

		fig_thresh = px.line(
			df, title='TPR and FPR at every threshold',
			width=700, height=500
		)
		fig_thresh.update_yaxes(scaleanchor="x", scaleratio=1)
		fig_thresh.update_xaxes(range=[0, 1], constrain='domain')
		fig_thresh.show()

	def plot_pr_curve(self) -> None:
		"""
		Using plotly to display the PR curve
		:return: None
		"""
		precision, recall, _ = precision_recall_curve(self.target, self.pred)
		fpr, tpr, _ = roc_curve(self.target, self.pred)
		fig = px.area(
			x=recall, y=precision,
			title=f'Precision-Recall Curve (AUC={auc(fpr, tpr):.4f})',
			labels=dict(x='Recall', y='Precision'),
			width=700, height=500
		)
		fig.add_shape(
			type='line', line=dict(dash='dash'),
			x0=0, x1=1, y0=1, y1=0
		)
		fig.update_yaxes(scaleanchor="x", scaleratio=1)
		fig.update_xaxes(constrain='domain')

		fig.show()
