import os
import location.wifi as wifi
import pandas as pd
import numpy as np

path = os.path.abspath(os.path.join(os.getcwd(), "./data"))
data_file = path + '/rssi_fluctuation/2020-07-10-07-37-32.csv'

df = pd.read_csv(data_file)
rssi = df[[col for col in df.columns if 'rssi' in col]].values

wifi = wifi.Model(rssi)
# wifi.rssi_fluctuation(False)
wifi.determineGaussian(rssi[:, 0], True, wipeRange=170*100)