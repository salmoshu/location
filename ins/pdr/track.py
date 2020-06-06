from PDR import PDR
import pandas as pd
import matplotlib.pyplot as plt

path = "../data/refer_data/19steps.csv"
df = pd.read_csv(path)
pdr = PDR(df['timestamp'].values,
          df['linear-x'].values,
          df['linear-y'].values,
          df['linear-z'].values,
          df['rotation-x'].values,
          df['rotation-y'].values,
          df['rotation-z'].values,
          df['rotation-w'].values)

z_acceleration, steps = pdr.pedometer()
print(len(steps))

index_test = []
value_test = []
for v in steps:
    index_test.append(v['index'])
    value_test.append(v['acceleration'])

plt.plot(z_acceleration)
plt.scatter(index_test, value_test, color='r')
plt.show()