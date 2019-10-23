import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import numpy as np
import seaborn as sns
import scipy.stats

sns.set_style("white")

train_data = pd.read_csv('datasets/train_interim_6.csv')

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

# HOUR VS DURATION
train_data.groupby("hour").DURATION.median().plot.bar()
show()

#DAY VS DURATION
train_data.groupby("dayofweek").DURATION.plot.kde()
show()
fig, ax = plt.subplots()
day_of_week_grp = train_data.groupby('dayofweek')
monday = day_of_week_grp.get_group(0)
tuesday = day_of_week_grp.get_group(1)
wednesday = day_of_week_grp.get_group(2)
thursday = day_of_week_grp.get_group(3)
friday = day_of_week_grp.get_group(4)
saturday = day_of_week_grp.get_group(5)
sunday = day_of_week_grp.get_group(6)

df = pd.DataFrame({
    'Monday': monday['DURATION'],
    'Tuesday': tuesday['DURATION'],
    'Wednesday': wednesday['DURATION'],
    'Thursday': thursday['DURATION'],
    'Friday': friday['DURATION'],
    'Saturday': saturday['DURATION'],
    'Sunday': sunday['DURATION']
})

kdea = scipy.stats.gaussian_kde(monday['DURATION'])
print(vars(kdea))

ax.plot(monday['DURATION'], kdea, color="crimson", lw=2, label = "pdf")
monday['DURATION'].plot.kde().set(xlim=(0, 200))
df.plot.kde().set(xlim=(0, 200))

show()

train_data['hour'].value_counts().plot(kind='bar',
                                   figsize=(14,8))

show()