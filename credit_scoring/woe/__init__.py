from ._woe_transform import WOE


def woe_binning(frame, target, combine, min_):
    return WOE(frame, target, min_, combine).woe_iv()





