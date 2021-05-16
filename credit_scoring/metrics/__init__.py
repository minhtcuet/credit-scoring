from ._ks import CalculatedKS
from ._lift import CalculatedLift
from ._roc_auc import CalculatedAUC


def ks(y_pred, y_target):
    return CalculatedKS(y_pred, y_target).ks()


def plot_ks_chart(y_pred, y_target):
    return CalculatedKS(y_pred, y_target).plot_ks_chart()


def lift(y_pred, y_target, bucket=10):
    return CalculatedLift(y_pred, y_target, bucket).lift()


def plot_gain_chart(y_pred, y_target, bucket=10):
    return CalculatedLift(y_pred, y_target, bucket).plot_gain_chart()


def plot_lift_chart(y_pred, y_target, bucket=10):
    return CalculatedLift(y_pred, y_target, bucket).plot_lift_chart()


def gini(y_pred, y_target):
    return CalculatedAUC(y_pred, y_target).gini()


def roc_auc(y_pred, y_target):
    return CalculatedAUC(y_pred, y_target).roc_auc()


def plot_roc_curve(y_pred, y_target):
    return CalculatedAUC(y_pred, y_target).plot_roc_curve()


def plot_compare_true_label(y_pred, y_target):
    return CalculatedAUC(y_pred, y_target).plot_compare_true_label()


def plot_tpr_fpr(y_pred, y_target):
    return CalculatedAUC(y_pred, y_target).plot_tpr_fpr()


def plot_pr_curve(y_pred, y_target):
    return CalculatedAUC(y_pred, y_target).plot_pr_curve()
