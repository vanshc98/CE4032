import numpy as np
import pandas as pd
from datetime import datetime
import pytz
import ast
import matplotlib.pyplot as plt
from utils import calHarDist, one_hot, heading, CC_LAT, CC_LON
import math
from tqdm import tqdm
import time
import statistics

starttime = time.time()
local_tz = pytz.timezone('Portugal')
train_data = pd.read_csv('datasets/train.csv')
holiday_data = pd.read_csv('datasets/Holidays.csv')

print("Removing Missing Data")
#only exist missing data in train data
for index, row in train_data.iterrows():
    if(row['MISSING_DATA']== True):
        train_data.drop(index, inplace = True)

train_data = train_data.reset_index(drop=True)
train_data = train_data.drop_duplicates(subset=['TRIP_ID'])
train_data = train_data.reset_index(drop=True)

print("Adding date")

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
holiday_test = holiday_data.copy()
print("Adding actual day type")

holiday_train = holiday_data.copy()
actual_day_type = []
for index, row in train_data.iterrows():
    taxi_datetime= row['DATE']
    taxi_datetime = taxi_datetime.replace(tzinfo=None)
    for index, row in holiday_train.iterrows():
        holiday_datetime = row['DateStamp']
        delta = taxi_datetime.date() - holiday_datetime.date()
        actual_day_type_value='A'
        if(delta.days > 5):
            holiday_train.drop(index, inplace = True)
        if(delta.days==0):
            actual_day_type_value = 'B'
            break
        elif(delta.days==-1):
            actual_day_type_value = 'C'
            break
        elif(delta.days<=-2):
            break
    actual_day_type.append(actual_day_type_value)

train_data['ACTUAL_DAYTYPE'] = actual_day_type
print(train_data.groupby('ACTUAL_DAYTYPE').count())

print("Adding duration")

trip_duration = []
end_time = []
origin = []
dest = []

for i in tqdm(range(train_data.shape[0])):
    try:
        polyline = train_data['POLYLINE'][i]
        if(polyline == '[]'):
            origin.append("NULL")
            dest.append("NULL")
            trip_duration.append("NULL")
            end_time.append("NULL")
        else:
            poly_arr = polyline.split('],[')
            dur = 15 * (len(poly_arr) - 1)
            trip_duration.append(dur)
            endtime = train_data['TIMESTAMP'][i] + dur
            end_time.append(endtime)
            origin.append(poly_arr[0][2:])
            dest.append(poly_arr[-1][:-2])
    except:
        origin.append("NULL")
        dest.append("NULL")
        trip_duration.append("NULL")
        end_time.append("NULL")

train_data['DURATION'] = trip_duration
train_data['ORIGIN'] = origin
train_data['DESTINATION'] = dest
train_data['END_TIME'] = end_time

day_of_week_list = []
hour_list = []

for i in range(train_data.shape[0]):
    ts = train_data['TIMESTAMP'][i]
    readable = datetime.fromtimestamp(ts) #convert to string time
    ts = pd.Timestamp(readable) #convert to pandas timestamp object
    day_of_week_list.append(ts.dayofweek)
    hour_list.append(ts.hour)

train_data['DAY_OF_WEEK'] = day_of_week_list
train_data['HOUR'] = hour_list

train_data = train_data.dropna(subset=['ORIGIN'])
for index, row in train_data.iterrows():
    if(row['ORIGIN']== "NULL"):
        train_data.drop(index, inplace = True)
train_data = train_data.reset_index(drop=True)

train_data['ORIGIN'] = train_data['ORIGIN'].str.replace(r']', '')
train_data['DESTINATION'] = train_data['DESTINATION'].str.replace(r'[', '')

## Splitting origin and destination for lat and lng columns
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

CC_LON = -8.615393063941816
CC_LAT = 41.15767687592546

origin_header = []
origin_distance_to_cc = []
for i in range(train_data.shape[0]):
    origin_lat = float(train_data['ORIGIN_LAT'][i])
    origin_lng = float(train_data['ORIGIN_LNG'][i])
    origin_header.append(heading((origin_lat, origin_lng), (CC_LAT, CC_LON)))
    origin_distance_to_cc.append(calHarDist(origin_lat, origin_lng, CC_LAT, CC_LON))

train_data['ORIGIN_HEADER'] = origin_header
train_data['ORIGIN_DISTANCE_TO_CC'] = origin_distance_to_cc

origin_cutoff_header = []
origin_distance_to_cutoff = []

for i in range(train_data.shape[0]):
    origin_lat = float(train_data['ORIGIN_LAT'][i])
    origin_lng = float(train_data['ORIGIN_LNG'][i])
    cutoff_lat = float(train_data['DEST_LNG'][i])
    cutoff_lng = float(train_data['DEST_LNG'][i])
    origin_cutoff_header.append(heading((origin_lat, origin_lng),(cutoff_lat, cutoff_lng)))
    origin_distance_to_cutoff.append(calHarDist(origin_lat, origin_lng, cutoff_lat, cutoff_lng))

train_data['HEADER_ORIGIN_TO_CUT_OFF'] = origin_cutoff_header
train_data['CUT_OFF_DIST_FROM_ORIGIN'] = origin_distance_to_cutoff

dist_list = []
median_velocity = []
final_velocity = []
train_data
for i in range(train_data.shape[0]):
    row = ast.literal_eval(train_data['POLYLINE'][i])
    cum_dist = 0
    temp = []
    temp_velocity = []
    for j in range(len(row) - 1):
        curr_lon = row[j][0]
        curr_lat = row[j][1]
        next_lon = row[j+1][0]
        next_lat = row[j+1][1]
        har_dist_travelled = calHarDist(curr_lat,curr_lon,next_lat,next_lon)
        cum_dist += har_dist_travelled
        temp_velocity.append(har_dist_travelled / 15)
    dist_list.append(cum_dist)
    if len(temp_velocity) != 0:
        median_velocity.append(statistics.median(temp_velocity))
        final_velocity.append(temp_velocity[-1])
    else:
        median_velocity.append(0)
        final_velocity.append(0)

train_data['CUM_DIST'] = dist_list
train_data['MEDIAN_VELOCITY'] = median_velocity
train_data['FINAL_VELOCITY'] = final_velocity

train_data.to_csv('datasets/modified_train_for_visualization.csv', index=False)
print("Completed in %s s" %(time.time()-starttime))

