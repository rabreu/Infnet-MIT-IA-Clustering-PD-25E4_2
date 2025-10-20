import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

url = "https://ocw.mit.edu/courses/15-097-prediction-machine-learning-and-statistics-spring-2012/89d88c5528513adc4002a1618ce2efb0_iris.csv"

data = pd.read_csv(url, header=None, names=['sepalLengthCm','sepalWidthCm','petalLengthCm','petalWidthCm', 'Species'])

print(data.head())

sns.scatterplot(data, x='sepalLengthCm', y='sepalWidthCm', hue='Species')
plt.show()
sns.scatterplot(data, x='petalLengthCm', y='petalWidthCm', hue='Species')
plt.show()

f, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

for i, column in enumerate(data.columns[0:4]):
  sns.histplot(data=data, x=column, hue='Species', ax=ax[i])

sns.despine()
plt.show()

profile = ProfileReport(data, title="Iris Dataframe")
profile.to_notebook_iframe()