import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from dateutil import relativedelta

"""
The equation resource
http://www.ndaa.org/pdf/toxicology_final.pdf

Grams of alcohol computation
(Volume of drinks) x (AC of drinks) x 0.789 = grams of alcohol consumed.

Where the following conversion factors are used:
1 fl. oz. = 29.6 mL
1 mL alcohol = 0.789 grams alcohol

1 drink is 14 grams of alcohol

num_drinks
drink_type
sex: Male, male, M, Female, female, F
weight: lbs, kg
timestamp: starting from 1990-01-01, move forward slowly 
bac: 

"""

#print(pd.read_csv('./person_data.csv', delimiter=';'))

def compute_bac(sample):
    if sample['sex'][0].upper() == "M":
        return sample['num_drinks']*14 / (sample['weight']*454*.68)*100

    elif sample['sex'][0].upper() == "F":
        return sample['num_drinks']*14 / (sample['weight']*454*.55)*100

def make_obs():
    sample = {}

    sample['collection_date'] = dt.datetime.today().isoformat()

    sample['sex'] = np.random.choice(['Male', 'M', 'male', 'meal', 'F', 'Female', 'femeal', 'female'])

    sample['age'] = np.random.uniform(21, 65)
    birthday = dt.datetime.now() - dt.timedelta(days=sample['age'] * 365 + np.random.uniform(0, 100))
    birthday = birthday.date()
    birthday_td = relativedelta.relativedelta(dt.date.today(), birthday)



    if sample['sex'][0].upper() == "M":
        sample['weight'] = round(np.random.normal(190, 20), 2)
    elif sample['sex'][0].upper() == "F":
        sample['weight'] = round(np.random.normal(165, 20), 2)

    sample['num_drinks'] = round(np.random.randint(1, 10)*(25/sample['age']), 1)

    if np.random.choice([True, False], 1):
        sample['age'] = birthday_td.years*12 + birthday_td.months
    else:
        sample['age'] = birthday.isoformat()


    sample['drink_type'] = np.random.choice(['Wine', 'Hard', 'Beer'])

    if sample['drink_type'] == 'Wine':
        sample['volume_consumed'] = 5*sample['num_drinks']
        sample['drink_type'] = np.random.choice(['Wine', 'Vino', 'Vin', 'Merlot', 'Cab'])

    elif sample['drink_type'] == 'Hard':
        sample['volume_consumed'] = 1.5*sample['num_drinks']
        sample['drink_type'] = np.random.choice(['Scotch', 'whiskey', 'vidka', 'shots'])

    elif sample['drink_type'] == 'Beer':
        sample['volume_consumed'] = 12*sample['num_drinks']
        sample['drink_type'] = np.random.choice(['Beer', 'Bud Light', 'Franziskaner Weissbier', 'IPA', 'Warsteiner'])

    sample['BAC'] = compute_bac(sample) + np.random.normal(0, 1)*.005


    # randomly change to metric units
    if np.random.choice([True, False], 1):
        sample['weight'] = sample['weight']*1.453592 + np.random.normal(0, 1)
        sample['volume_consumed'] = sample['volume_consumed']*29.6 + np.random.normal(0, 1)
        sample['units'] = np.random.choice(['SI', 'metric', 'metric'])

    # Randomly drop Sex
    if np.random.choice([True, False], 1, p=[.1, .9]):
        sample['sex'] = None


    return sample

data = [make_obs() for i in range(2000)]
data = pd.DataFrame.from_records(data)
data.to_csv('./horrible_data.csv')

data.plot.scatter(x='num_drinks', y='BAC')
plt.show()
