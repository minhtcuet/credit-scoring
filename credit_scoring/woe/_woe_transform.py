import pandas as pd
import numpy as np
from collections import defaultdict


class WOE:
    """
    Process the WOE-IV for dataframe
    """

    def __init__(self, frame: pd.DataFrame, target: str,
                 min_unique_to_combine: int, is_combine=False):
        """

        :param frame: the dataframe contains features (only features need to be woe) and target (output)
        :param target: the label
        """
        self.frame = frame
        self.target = target
        self.is_combine = is_combine
        self.min_unique_to_combine = min_unique_to_combine
        self.check_type()

    def check_type(self):
        """
        Check valid input
        :return:
        """
        # the first input must be dataframe
        if not isinstance(self.frame, pd.DataFrame):
            msg_error = "The first input must be dataframe"
            raise ValueError(msg_error)

        # check the target label is in input dataframe
        if self.target not in self.frame.columns:
            msg_error = "The target label '{}' must be in input data-frame".format(self.target)
            raise ValueError(msg_error)

        # check the target label is binary type
        if self.frame[self.target].nunique() != 2:
            msg_error = "The label target is not binary type."
            raise ValueError(msg_error)

        # check the target only contains 0 and 1,
        if sorted(list(self.frame[self.target].unique())) != [0, 1]:
            msg_error = "The label target should be encode to 0 and 1"
            raise ValueError(msg_error)

    def mono_woe(self, column):
        """
        Calculate woe for 1 features
        :return: woe table value
        """

        res = list()
        values = self.frame[column].unique()
        for value in values:
            tmp = self.frame[self.frame[column] == value]
            res.append(
                {
                    'Value': value,
                    'Occur': tmp.shape[0],
                    'Event': tmp.loc[tmp[self.target] == 1].shape[0],
                    'NonEvent': tmp.loc[tmp[self.target] == 0].shape[0],
                }
            )
        data = pd.DataFrame(res)
        data['Good_Distribution'] = data['NonEvent'] / data['NonEvent'].sum()
        data['Bad_Distribution'] = data['Event'] / data['Event'].sum()

        # WOE = Ln(% non-event / % event)
        data['WOE'] = np.log(data['Good_Distribution'] / data['Bad_Distribution'])

        # Replace infinity to 0, sometime the bad distribution equal 0
        data = data.replace({'WOE': {np.inf: 0, -np.inf: 0}})

        # IV = (% non-event - %event) * woe
        data['IV'] = (data['Good_Distribution'] - data['Bad_Distribution']) * data['WOE']
        iv = data['IV'].sum()
        data = data.sort_values(by=['WOE'])

        # check combine flag
        group = None
        if self.is_combine and self.min_unique_to_combine < len(values):
            data, group = self.auto_combined(data, self.min_unique_to_combine)

        data['Feature'] = column
        return data, iv, group

    @staticmethod
    def auto_combined(frame, q):
        min_, max_ = frame['WOE'].min() - 1e-3, frame['WOE'].max() + 1e-3
        range_ = np.linspace(min_, max_, q + 1)
        frame['TMP'] = pd.cut(frame['WOE'], range_)
        values = frame['TMP'].unique()
        res, group = list(), defaultdict(set)

        for idx, value in enumerate(values):
            k = frame.loc[frame['TMP'] == value]
            k1 = pd.DataFrame(np.mean(k)).T
            name_group = 'GROUP_{}'.format(idx)
            k1['Value'] = name_group
            res.append(k1)
            group[name_group] |= set(k['Value'].unique())

        res = pd.concat(res)
        res = res.sort_values(by=['WOE'], ascending=False).reset_index(drop=True)
        return res, group

    def woe_iv(self):
        columns = set(self.frame.columns) ^ {self.target}
        woe, iv = list(), dict()

        # Replace NA value with String to handle na case
        self.frame.fillna("NA Value", inplace=True)

        res = {}
        for column in columns:
            t, r, _ = self.mono_woe(column)
            woe.append(t)
            iv[column] = r
            res[column] = _

        iv = pd.DataFrame([iv]).T
        woe = pd.concat(woe)
        return woe, iv, res
