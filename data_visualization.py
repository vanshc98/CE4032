import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import show
import numpy as np
import seaborn as sns
import scipy.stats

labels = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
sns.set_style("white")

train_data = pd.read_csv('datasets/modified_train_for_visualization.csv')

# ORIGIN_STAND VS DURATION
train_data["ORIGIN_STAND"] = train_data["ORIGIN_STAND"].apply(lambda x: x if not np.isnan(x) else -1)
ax = train_data.groupby("ORIGIN_STAND").DURATION.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Duration(s) v/s Origin Stand')
ax.axis(xmin=-2)
ax.set_xlabel('Origin Stand')
ax.set_ylabel('Duration(s)')
show()

# # TAXI_ID VS DURATION
ax = train_data.groupby("TAXI_ID").DURATION.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Duration(s) v/s Taxi ID')
ax.set_xlabel('Taxi ID')
ax.set_ylabel('Duration(s)')
show()

# ACTUAL_DAYTYPE vs DURATION
ax = train_data.groupby("ACTUAL_DAYTYPE").DURATION.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Duration(s) v/s Actual Day Type')
ax.set_xlabel('Actual Day Type')
ax.set_ylabel('Duration(s)')
show()

# CALL_TYPE VS DURATION
ax = train_data.groupby("CALL_TYPE").DURATION.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Duration(s) v/s Call Type')
ax.set_xlabel('Call Type')
ax.set_ylabel('Duration(s)')
show()


# #DAY VS DURATION
fig, ax = plt.subplots()
day_of_week_grp = train_data.groupby('DAY_OF_WEEK')
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

df.plot.kde().set(xlim=(0, 4000))

show()

# # To plot number of pickups for different times of the day
ax = train_data['HOUR'].value_counts(sort = False).plot(kind='line', marker = 'o', color = 'b', title = 'Number of Pick-ups vs Hours of the day',
                                                  figsize=(14,8))
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Number of Pick-ups')
show()

# # To plot number of pickups on different days and different times of the day
df2 = pd.DataFrame({
    'Monday': monday['HOUR'].value_counts(sort = False),
    'Tuesday': tuesday['HOUR'].value_counts(sort = False),
    'Wednesday': wednesday['HOUR'].value_counts(sort = False),
    'Thursday': thursday['HOUR'].value_counts(sort = False),
    'Friday': friday['HOUR'].value_counts(sort = False),
    'Saturday': saturday['HOUR'].value_counts(sort = False),
    'Sunday': sunday['HOUR'].value_counts(sort = False)
})
ax = df2.plot(kind='line', marker = 'o', linestyle = '--', figsize=(14,8), title = 'Pick-ups vs Hours of the day (Days of the week)')
ax.set_xlabel('Hour of the day')
ax.set_ylabel('Number of pickups')
show()

# # Median Velocity v/s time of the day
ax1 = train_data.groupby("HOUR").MEDIAN_VELOCITY.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Median Velocity(m/s) v/s Time of the Day')
ax1.set_ylabel('Median Velocity(m/s)')
ax1.set_xlabel('Hour')
show()

# # Cumulative Distance v/s time of the day
ax2 = train_data.groupby("HOUR").CUM_DIST.median().plot.bar(figsize=(14,8), title = 'Distance Travelled(m) v/s Time of the Day')
ax2.set_ylabel('Distance Travelled(m)')
ax2.set_xlabel('Hour')
show()
ax2 = train_data.groupby("HOUR").CUM_DIST.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Distance Travelled(m) v/s Time of the Day')
ax2.set_ylabel('Distance Travelled(m)')
ax2.set_xlabel('Hour')
show()

# # Duration v/s time of the day
ax3 = train_data.groupby("HOUR").DURATION.median().plot(kind='line', marker = 'o', color = 'b', figsize=(14,8), title = 'Duration(s) v/s Time of the Day')
ax3.set_ylabel('Duration(s)')
ax3.set_xlabel('Hour')
show()

# Median Velocoty v/s day of the week
ax1 = train_data.groupby("DAY_OF_WEEK").MEDIAN_VELOCITY.median().plot.line(figsize=(14,8), marker = 'o', title = 'Median Velocity(m/s) vs Day of the Week')
ax1.set_xlabel('Day of the week')
ax1.set_ylabel('Median Velocity(m/s)')
ax1.set_xticklabels(labels, rotation = 45)
show()

# Cumulative Distance v/s day of the week
ax2 = train_data.groupby("DAY_OF_WEEK").CUM_DIST.median().plot.line(figsize=(14,8), marker = 'o', title = "Distance Travelled(m) vs Day of the Week")
ax2.set_xlabel('Day of the week')
ax2.set_ylabel('Distance Travelled(m)')
ax2.set_xticklabels(labels, rotation = 45)
show()

# Duration v/s day of the week
ax3 = train_data.groupby("DAY_OF_WEEK").DURATION.median().plot.line(figsize=(14,8), marker = 'o', title = "Time Travelled(s) vs Day of the Week")
ax3.set_xlabel('Day of the week')
ax3.set_ylabel('Time Travelled(s)')
ax3.set_xticklabels(labels, rotation = 45)
show()