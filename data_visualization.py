import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import numpy as np
import seaborn as sns
import scipy.stats

labels = ['','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
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

df.plot.kde().set(xlim=(0, 200))

show()

# To plot number of pickups for different times of the day
ax = train_data['hour'].value_counts(sort = False).plot(kind='line', marker = 'o', color = 'b', title = 'Number of Pick-ups vs Hours of the day',
                                                  figsize=(14,8))
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Number of Pick-ups')
ax.plot()

# To plot number of pickups on different days and different times of the day
df2 = pd.DataFrame({
    'Monday': monday['hour'].value_counts(sort = False),
    'Tuesday': tuesday['hour'].value_counts(sort = False),
    'Wednesday': wednesday['hour'].value_counts(sort = False),
    'Thursday': thursday['hour'].value_counts(sort = False),
    'Friday': friday['hour'].value_counts(sort = False),
    'Saturday': saturday['hour'].value_counts(sort = False),
    'Sunday': sunday['hour'].value_counts(sort = False)
})
ax = df2.plot(kind='line', marker = 'o', linestyle = '--', figsize=(14,8), title = 'Hours of the day (Days of the week) vs Pick-ups')
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Number of pickups')
ax.plot()

# Median Velocity v/s time of the day
ax1 = train_data.groupby("hour").MEDIAN_VELOCITY.median().plot.bar(figsize=(14,8), title = 'Median Velocity v/s Time of the Day')
ax1.set_ylabel('Median Velocity')
ax1.set_xlabel('Hour')
show()

# Cumulative Distance v/s time of the day
ax2 = train_data.groupby("hour").CUM_DIST.median().plot.bar(figsize=(14,8), title = 'Distance Travelled v/s Time of the Day')
ax2.set_ylabel('Distance Travelled')
ax2.set_xlabel('Hour')
show()

# Duration v/s time of the day
ax3 = train_data.groupby("hour").DURATION.median().plot.bar(figsize=(14,8), title = 'Duration v/s Time of the Day')
ax3.set_ylabel('Duration')
ax3.set_xlabel('Hour')
show()

# Median Velocoty v/s day of the week
ax1 = train_data.groupby("dayofweek").MEDIAN_VELOCITY.median().plot.line(marker = 'o', title = 'Median Velocity vs Day of the Week')
ax1.set_xlabel('Day of the week')
ax1.set_ylabel('Median Velocity')
ax1.set_xticklabels(labels, rotation = 90)
show()

# Cumulative Distance v/s day of the week
ax2 = train_data.groupby("dayofweek").CUM_DIST.median().plot.line(marker = 'o')
ax2.set_xlabel('Day of the week')
ax2.set_ylabel('Distance Travelled')
ax2.set_xticklabels(labels, rotation = 90)
show()

# Duration v/s day of the week
ax3 = train_data.groupby("dayofweek").DURATION.median().plot.line(marker = 'o')
ax3.set_xlabel('Day of the week')
ax3.set_ylabel('Time Travelled')
ax3.set_xticklabels(labels, rotation = 90)
show()
