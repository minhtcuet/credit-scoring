## Project Description

#### Credit-Score

Credit Score is a Python module for credit scoring problems, help developers/ researchers applied woe-iv for easier.
More, It provides some importance reports about credit scoring contains: GINI, KS, LIFT, GAIN, IRL, LR, ... help users 
save time to choose the best model.

Currently, credit-scoring package handles only binary target

#### Installation

This package requires numpy, pandas, sklearn and plotly. We can use this package for Python 3, and pandas must be >= 1.0 

To install the package, run this command:

`python setup.py install `

or 

`pip install credit_scoring`

If you want install the development of this package, you can visit: 
_`https://github.com/minhtcuet/creditscoring`_

#### Usage 

This version, we focus on reporting first. You can have quick GINI, KS, LIFT, .... with 2 input parameters contains: 
y_target(binary type) and y_predict(probability)

#### Example

###### GINI Calculated

from credit_scoring.AUC import GINI

gini = GINI(y_predict, y_label)
print(gini)_


