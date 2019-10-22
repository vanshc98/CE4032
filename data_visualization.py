import pandas as pd
from matplotlib.pyplot import show
import numpy as np

train_data = pd.read_csv('data/train with duration origin etc v4.csv')

# ORIGIN_STAND VS DURATION
train_data["ORIGIN_STAND"] = train_data["ORIGIN_STAND"].apply(lambda x: x if not np.isnan(x) else -1)
train_data.groupby("ORIGIN_STAND").DURATION.median().plot.bar()
show()

# TAXI_ID VS DURATION
train_data.groupby("TAXI_ID").DURATION.median().plot.bar()
show()

# ACTUAL_DAYTYPE VS DURATION
train_data.groupby("ACTUAL_DAYTYPE").DURATION.median().plot.bar()
show()

# CALL_TYPE VS DURATION
train_data.groupby("CALL_TYPE").DURATION.median().plot.bar()
show()