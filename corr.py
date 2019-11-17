import pandas as pd
import seaborn as sns
import pylab as plt

train_data = pd.read_csv('datasets/modified_train.csv')

train_data = train_data.drop(columns=['END_TIME'])

cors = train_data.corr()
plt.figure(figsize=(32,24))
sns.heatmap(cors, annot=False ,cmap = 'viridis')
plt.savefig('./pictures/corr.png')
plt.show()
