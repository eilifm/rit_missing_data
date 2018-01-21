import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA

#age = np.random.triangular(18, 35, 65, 1000)
nsample = 100
data = pd.DataFrame()
data['Sex'] = np.random.choice(['M', 'F'], size=nsample)
data['WrkXPYrs'] = np.random.triangular(1, 15, 25, size=nsample)
data['OIX'] = np.random.exponential(50000, size=nsample) + 10000
data['EDU'] = np.random.triangular(6, 16, 22, size=nsample)
data['Luck'] = np.random.gamma(2, .5, size=nsample)*10
data['Luck'] = data['Luck']/(data['Luck'].max() - data['Luck'].min())*100
data = data.join(pd.get_dummies(data['Sex'], prefix='Sex'))

del data['Sex']
del data['Sex_F']

# Add the interaction terms
data['Sex_M:Luck'] = data['Sex_M']*data['Luck']

# Declare coefficients
beta = np.array([1, 10, .1, 100, 10, 2, -5])*10
e = np.random.normal(size=nsample)
data = sm.add_constant(data)

# Generate response variable
data['Income'] = np.dot(data.values, beta) + e

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
vif["features"] = data.columns

print(vif)

print(data.corr())
#data['Sex:EDU'] = X['Sex_M']*X['EDU']
print(data)

data.to_csv('sim_data.csv', index=False)



X = data.loc[0:10, data.columns != "Income"]
y = data.loc[0:10, 'Income']


model = sm.OLS(y, X)
results = model.fit()
print(results.summary())



#income = 50000 + (1000*years_working) + (1.76*male) + (50*yrs_education) + np.random.rand(1000)
