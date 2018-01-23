import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

nsample = 100
data = pd.DataFrame()
data['x1'] = np.random.lognormal(0, 1, size=nsample)
data['x2'] = np.random.normal(10, 1, size=nsample)


# Declare coefficients
betas = {}
betas['x1'] = 2
betas['x2'] = 2

beta_list = [betas[x] for x in data.columns.values]

dotted = np.dot(data, beta_list) + 5

data['y'] = dotted + np.random.normal(0, .1*np.mean(dotted), size=nsample)




# vif = pd.DataFrame()
# vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
# vif["features"] = data.columns


#
data.to_csv('sim_data.csv', index=False)
#
#
#
X = data.loc[:, data.columns != "y"]
y = data.loc[:, 'y']
#

X = sm.add_constant(X)
model = sm.OLS(y, X, hasconst=False)
results = model.fit()

print(results.summary())
#
#
#
# #income = 50000 + (1000*years_working) + (1.76*male) + (50*yrs_education) + np.random.rand(1000)
