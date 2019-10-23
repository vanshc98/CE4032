import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import ast
import matplotlib.pyplot as plt
from utils import calHarDist, one_hot

local_tz = pytz.timezone('Portugal')
test_data = pd.read_csv('datasets/test.csv')
train_data = pd.read_csv('datasets/train.csv')

#train_data.head()
#train_data.groupby('MISSING_DATA').count()

holiday_data = pd.read_csv('datasets/Holidays.csv')
#holiday_data.head()

print("Removing Missing Data")
#only exist missing data in train data
for index, row in train_data.iterrows():
    if(row['MISSING_DATA']== True):
        train_data.drop(index, inplace = True)

train_data = train_data.reset_index(drop=True)

print("Adding date")
#add datestamp to test data
datestamp_list = []

for index, row in test_data.iterrows():
    epoch_datestamp = row['TIMESTAMP']
    datetime_for_datestamp_in_utc = datetime.utcfromtimestamp(epoch_datestamp)
    actual_datetime_for_datestamp = datetime_for_datestamp_in_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
    datestamp_list.append(actual_datetime_for_datestamp)

test_data['DATE'] = datestamp_list

#add datestamp to train data
datestamp_list = []

for index, row in train_data.iterrows():
    epoch_datestamp = row['TIMESTAMP']
    datetime_for_datestamp_in_utc = datetime.utcfromtimestamp(epoch_datestamp)
    actual_datetime_for_datestamp = datetime_for_datestamp_in_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
    datestamp_list.append(actual_datetime_for_datestamp)

train_data['DATE'] = datestamp_list

holiday_date=[]
for index, row in holiday_data.iterrows():
        holiday_datestamp = row['Date']
        holiday_datestamp = holiday_datestamp.replace("\t", "")
        holiday_datetime = datetime.strptime(holiday_datestamp, '%d.%m.%Y')
        holiday_date.append(holiday_datetime)

holiday_data['DateStamp']=holiday_date

print("Adding actual day type")

#for test data
actual_day_type = []
counter = 0
for index, row in test_data.iterrows():
    taxi_datetime= row['DATE']
    taxi_datetime = taxi_datetime.replace(tzinfo=None)
    flag=0
    for index, row in holiday_data.iterrows():
        holiday_datetime = row['DateStamp']
        delta = taxi_datetime.date() - holiday_datetime.date()
        actual_day_type_value='A'
        if(delta.days > 5):
            holiday_data.drop(index, inplace = True)
        if(delta.days==0):
            actual_day_type_value = 'B'
            break
        elif(delta.days==-1):
            actual_day_type_value = 'C'
            break
        elif(delta.days<=-2):
            break
    actual_day_type.append(actual_day_type_value)
    counter+=1
    if(counter%500==0):
        print(counter/1710670)
test_data['ACTUAL_DAYTYPE'] = actual_day_type

#for train data
actual_day_type = []
counter = 0
for index, row in train_data.iterrows():
    taxi_datetime= row['DATE']
    taxi_datetime = taxi_datetime.replace(tzinfo=None)
    flag=0
    for index, row in holiday_data.iterrows():
        holiday_datetime = row['DateStamp']
        delta = taxi_datetime.date() - holiday_datetime.date()
        actual_day_type_value='A'
        if(delta.days > 5):
            holiday_data.drop(index, inplace = True)
        if(delta.days==0):
            actual_day_type_value = 'B'
            break
        elif(delta.days==-1):
            actual_day_type_value = 'C'
            break
        elif(delta.days<=-2):
            break
    actual_day_type.append(actual_day_type_value)
    counter+=1
    if(counter%500==0):
        print(counter/1710670)
train_data['ACTUAL_DAYTYPE'] = actual_day_type
train_data.groupby('ACTUAL_DAYTYPE').count()

print("Adding duration")

#for test data

trip_duration = []
end_time = []
origin = []
dest = []

for i in range(test_data.shape[0]):
    try:
        polyline = test_data['POLYLINE'][i]
        if(polyline==[]):
            origin.append("NULL")
            dest.append("NULL")
            trip_duration.append("NULL")
            end_time.append("NULL")
        else:
            poly_arr = polyline.split('],[')
            dur = len(poly_arr)
            trip_duration.append(dur)
            endtime = train_data['TIMESTAMP'][i] + (dur * 15)
            end_time.append(endtime)
            origin.append(poly_arr[0][2:])
            dest.append(poly_arr[-1][:-2])
    except:
        origin.append("NULL")
        dest.append("NULL")
        trip_duration.append("NULL")
        end_time.append("NULL")
    #if (i%100000) == 0:
    #    print(i)

test_data['DURATION'] = trip_duration
test_data['ORIGIN'] = origin
test_data['DESTINATION'] = dest
test_data['END_TIME'] = end_time

test_data = test_data.dropna(subset=['ORIGIN'])
test_data = test_data.reset_index(drop=True)

test_data['ORIGIN'] = test_data['ORIGIN'].str.replace(r']', '')
test_data['DESTINATION'] = test_data['DESTINATION'].str.replace(r'[', '')

#for train data

trip_duration = []
end_time = []
origin = []
dest = []

for i in range(train_data.shape[0]):
    try:
        polyline = train_data['POLYLINE'][i]
        if(polyline==[]):
            origin.append("NULL")
            dest.append("NULL")
            trip_duration.append("NULL")
            end_time.append("NULL")
        else:
            poly_arr = polyline.split('],[')
            dur = len(poly_arr)
            trip_duration.append(dur)
            endtime = train_data['TIMESTAMP'][i] + (dur * 15)
            end_time.append(endtime)
            origin.append(poly_arr[0][2:])
            dest.append(poly_arr[-1][:-2])
    except:
        origin.append("NULL")
        dest.append("NULL")
        trip_duration.append("NULL")
        end_time.append("NULL")
    #if (i%100000) == 0:
    #    print(i)

train_data['DURATION'] = trip_duration
train_data['ORIGIN'] = origin
train_data['DESTINATION'] = dest
train_data['END_TIME'] = end_time

train_data = train_data.dropna(subset=['ORIGIN'])
train_data = train_data.reset_index(drop=True)

train_data['ORIGIN'] = train_data['ORIGIN'].str.replace(r']', '')
train_data['DESTINATION'] = train_data['DESTINATION'].str.replace(r'[', '')


# get day of week and hour

print("Adding day of week and hour")

#for test_data
day_of_week_list = []
hour_list = []

for i in range(test_data.shape[0]):
    ts = test_data['TIMESTAMP'][i]
    ts = datetime.utcfromtimestamp(ts)
    actual_datetime_for_datestamp = ts.replace(tzinfo=pytz.utc).astimezone(local_tz)
    day_of_week_list.append(actual_datetime_for_datestamp.weekday())
    hour_list.append(actual_datetime_for_datestamp.hour)

test_data['dayofweek'] = day_of_week_list
test_data['hour'] = hour_list

#for train_data
day_of_week_list = []
hour_list = []

for i in range(train_data.shape[0]):
    ts = train_data['TIMESTAMP'][i]
    readable = datetime.fromtimestamp(ts) #convert to string time
    ts = pd.Timestamp(readable) #convert to pandas timestamp object
    day_of_week_list.append(ts.dayofweek)
    hour_list.append(ts.hour)

train_data['dayofweek'] = day_of_week_list
train_data['hour'] = hour_list

#get harversine distance function

#for test

print("Adding cumulative distance")

dist_list = []

for i in range(test_data.shape[0]):
    row = ast.literal_eval(test_data['POLYLINE'][i])
    cum_dist = 0
    temp = []
    for j in range(len(row)-1):
        curr_lon = row[j][0]
        curr_lat = row[j][1]
        next_lon = row[j+1][0]
        next_lat = row[j+1][1]

        har_dist_travelled = calHarDist(curr_lat,curr_lon,next_lat,next_lon)
        cum_dist += har_dist_travelled

    dist_list.append(cum_dist)

test_data['cum_dist'] = dist_list

#for train
dist_list = []

for i in range(train_data.shape[0]):
    row = ast.literal_eval(train_data['POLYLINE'][i])
    cum_dist = 0
    temp = []
    for j in range(len(row)-1):
        curr_lon = row[j][0]
        curr_lat = row[j][1]
        next_lon = row[j+1][0]
        next_lat = row[j+1][1]
        har_dist_travelled = calHarDist(curr_lat,curr_lon,next_lat,next_lon)
        cum_dist += har_dist_travelled
    dist_list.append(cum_dist)

train_data['cum_dist'] = dist_list

for index, row in train_data.iterrows():
    if(row['MISSING_DATA']== True):
        train_data.drop(index, inplace = True)

## Splitting origin and destination for lat and lng columns
# For Test Data

print("Getting Origin and Destination Lat Lng Coordinates")

# for test data
ox = []
oy = []
dx = []
dy = []

for i in range(test_data.shape[0]):
    origin = test_data['ORIGIN'][i].split(',')
    ox.append(origin[0])
    oy.append(origin[1])
    dest = test_data['DESTINATION'][i].split(',')
    dx.append(dest[0])
    dy.append(dest[1])

test_data['ORIGIN_LNG'] = ox
test_data['ORIGIN_LAT'] = oy
test_data['DEST_LNG'] = dx
test_data['DEST_LAT'] = dy

# For Train Data

ox = []
oy = []
dx = []
dy = []

for i in range(train_data.shape[0]):
    origin = train_data['ORIGIN'][i].split(',')
    ox.append(origin[0])
    oy.append(origin[1])
    dest = train_data['DESTINATION'][i].split(',')
    dx.append(dest[0])
    dy.append(dest[1])

train_data['ORIGIN_LNG'] = ox
train_data['ORIGIN_LAT'] = oy
train_data['DEST_LNG'] = dx
train_data['DEST_LAT'] = dy

print("Adding One-hot Encoding")

# for test data
one_hot_call_type = one_hot(test_data,0)
one_hot_day_type  = one_hot(test_data,1)
test_data = test_data.join(one_hot_call_type)
test_data = test_data.join(one_hot_day_type)

# for train data
one_hot_call_type = one_hot(train_data,0)
one_hot_day_type  = one_hot(train_data,1)
train_data = train_data.join(one_hot_call_type)
train_data = train_data.join(one_hot_day_type)

print("Writing dataframe to CSV")

train_data.to_csv('datasets/train_latest.csv', index=False)
