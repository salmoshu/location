import os
from ANALYSIS import ANALYSIS
import pandas as pd
import numpy as np

path = os.path.abspath(os.path.join(os.getcwd(), "../experiment_data"))
data_file = path + '/Gaussian/2020-07-10-07-37-32.csv'

df = pd.read_csv(data_file)
rssi = df[[col for col in df.columns if 'rssi' in col]].values

analysis = ANALYSIS()
analysis.rssi_fluctuation(rssi, False)
# analysis.determineGaussian(rssi[:, 0], True, wipeRange=170*100)