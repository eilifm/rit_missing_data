
import pandas as pd
import re
import matplotlib.pyplot as plt


in_data = pd.read_csv('./horrible_data.csv')
del in_data['Unnamed: 0']


print(in_data.loc[
          in_data['units'].isin(('metric', 'SI')), ['volume_consumed']])

print(in_data.dtypes)
