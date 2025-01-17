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

print("Reading Test and Train Data")
MAX_SAMPLES_PER_TRIP = 3
local_tz = pytz.timezone('Portugal')
test_data = pd.read_csv('datasets/test.csv')
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
holiday_test = holiday_data.copy()
print("Adding actual day type")

#for test data
actual_day_type = []
for index, row in test_data.iterrows():
    taxi_datetime= row['DATE']
    taxi_datetime = taxi_datetime.replace(tzinfo=None)
    for index, row in holiday_test.iterrows():
        holiday_datetime = row['DateStamp']
        delta = taxi_datetime.date() - holiday_datetime.date()
        actual_day_type_value='A'
        if(delta.days > 5):
            holiday_test.drop(index, inplace = True)
        if(delta.days==0):
            actual_day_type_value = 'B'
            break
        elif(delta.days==-1):
            actual_day_type_value = 'C'
            break
        elif(delta.days<=-2):
            break
    actual_day_type.append(actual_day_type_value)

test_data['ACTUAL_DAYTYPE'] = actual_day_type
print(test_data.groupby('ACTUAL_DAYTYPE').count())

#for train data
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

#for test data

trip_duration = []
end_time = []
origin = []
dest = []

for i in range(test_data.shape[0]):
    try:
        polyline = test_data['POLYLINE'][i]
        if(polyline=='[]'):
            origin.append("NULL")
            dest.append("NULL")
            trip_duration.append("NULL")
            end_time.append("NULL")
        else:
            poly_arr = polyline.split('],[')
            dur = len(poly_arr)
            trip_duration.append(dur)
            total_dur = 15 * (dur - 1)
            endtime = test_data['TIMESTAMP'][i] + total_dur
            end_time.append(endtime)
            origin.append(poly_arr[0][2:])
            dest.append(poly_arr[-1][:-2])
    except:
        origin.append("NULL")
        dest.append("NULL")
        trip_duration.append("NULL")
        end_time.append("NULL")

test_data['CUT_OFF_LENGTH'] = trip_duration
test_data['ORIGIN'] = origin
test_data['DESTINATION'] = dest
test_data['END_TIME'] = end_time

test_data = test_data.dropna(subset=['ORIGIN'])
for index, row in test_data.iterrows():
    if(row['ORIGIN']== "NULL"):
        test_data.drop(index, inplace = True)
test_data = test_data.reset_index(drop=True)

test_data['ORIGIN'] = test_data['ORIGIN'].str.replace(r']', '')
test_data['DESTINATION'] = test_data['DESTINATION'].str.replace(r'[', '')

#for train data

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

train_data = train_data.dropna(subset=['ORIGIN'])
for index, row in train_data.iterrows():
    if(row['ORIGIN']== "NULL"):
        train_data.drop(index, inplace = True)
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

test_data['DAY_OF_WEEK'] = day_of_week_list
test_data['HOUR'] = hour_list

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

# # Calculating city centre
print("Getting city centre coordinates")

X_coord = []
Y_coord = []
Z_coord = []
for index, row in tqdm(train_data.iterrows()):
    lat1 = ast.literal_eval(row['DESTINATION'])[0]
    lon1 = ast.literal_eval(row['DESTINATION'])[1]
    lat1 = lat1*(math.pi)/180
    lon1 = lon1*(math.pi)/180
    X = math.cos(lat1)*math.cos(lon1)
    Y = math.cos(lat1)*math.sin(lon1)
    Z = math.sin(lat1)
    X_coord.append(X)
    Y_coord.append(Y)
    Z_coord.append(Z)

x = np.median(X_coord)
y = np.median(Y_coord)
z = np.median(Z_coord)
Lon = math.atan2(y,x)
hyp = math.sqrt(x*x + y*y)
Lat = math.atan2(z,hyp)
Lat_city_center = Lat*180./(math.pi)
Lon_city_center = Lon*180./(math.pi)

# CC_LON = Lon_city_center
# CC_LAT = Lat_city_center
CC_LON = -8.615393063941816
CC_LAT = 41.15767687592546

#Calculating origin header and origin distance to city centre
print("Getting origin header and origin distance to city centre")
# for test data
origin_header = []
origin_distance_to_cc = []
for i in range(test_data.shape[0]):
    origin_lat = float(test_data['ORIGIN_LAT'][i])
    origin_lng = float(test_data['ORIGIN_LNG'][i])
    origin_header.append(heading((origin_lat, origin_lng), (CC_LAT, CC_LON)))
    origin_distance_to_cc.append(calHarDist(origin_lat, origin_lng, CC_LAT, CC_LON))

test_data['ORIGIN_HEADER'] = origin_header
test_data['ORIGIN_DISTANCE_TO_CC'] = origin_distance_to_cc
# for train data
origin_header = []
origin_distance_to_cc = []
for i in range(train_data.shape[0]):
    origin_lat = float(train_data['ORIGIN_LAT'][i])
    origin_lng = float(train_data['ORIGIN_LNG'][i])
    origin_header.append(heading((origin_lat, origin_lng), (CC_LAT, CC_LON)))
    origin_distance_to_cc.append(calHarDist(origin_lat, origin_lng, CC_LAT, CC_LON))

train_data['ORIGIN_HEADER'] = origin_header
train_data['ORIGIN_DISTANCE_TO_CC'] = origin_distance_to_cc

# Getting one-hot encoded call type and actual day type
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

print("Applying n sample on train data and expand test data")
# n_sample
def process_row_training(X, row):
    pln = ast.literal_eval(row['POLYLINE'])
    if len(pln)>3:
        n_samples = MAX_SAMPLES_PER_TRIP
        for i in range(n_samples):
            idx = np.random.randint(len(pln)-1) + 1
            if idx < 4:
                continue
            data = [row['TRIP_ID'], row['ORIGIN_CALL'], row['ORIGIN_STAND'], row['TAXI_ID'], row['TIMESTAMP'], row['DATE'], row['END_TIME'], row['dayofweek'], row['hour'], row['ORIGIN_LNG'], row['ORIGIN_LAT'], row['DEST_LNG'], row['DEST_LAT'], row['ORIGIN_HEADER'], row['ORIGIN_DISTANCE_TO_CC']]
            data += [idx, pln[idx][1], pln[idx][0], calHarDist(pln[idx][1], pln[idx][0], CC_LAT, CC_LON), heading([CC_LAT, CC_LON], pln[idx])]
            data += [row['CALL_TYPE_A'], row['CALL_TYPE_B'], row['CALL_TYPE_C'], row['ACTUAL_DAYTYPE_A'], row['ACTUAL_DAYTYPE_B'], row['ACTUAL_DAYTYPE_C'], row['DURATION']]
            X.append(data)
    return X

X = []
for i in range(train_data.shape[0]):
    X = process_row_training(X, train_data.iloc[i])

modified_train = pd.DataFrame.from_records(X)
modified_train.columns = ['TRIP_ID', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DATE', 'END_TIME', 'DAY_OF_WEEK', 'HOUR', 'ORIGIN_LNG', 'ORIGIN_LAT', 'DEST_LNG', 'DEST_LAT', 'ORIGIN_HEADER', 'ORIGIN_DISTANCE_TO_CC', 'CUT_OFF_LENGTH', 'CUT_OFF_LAT', 'CUT_OFF_LNG', 'CUT_OFF_DIST_FROM_CC', 'HEADER_CUT_OFF_TO_CC', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'ACTUAL_DAYTYPE_A', 'ACTUAL_DAYTYPE_B', 'ACTUAL_DAYTYPE_C', 'DURATION']

# testing data cutoff to CC and header to CC
cutoff_header = []
cutoff_distance_to_cc = []

for i in range(test_data.shape[0]):
    cutoff_lat = float(test_data['DEST_LAT'][i])
    cutoff_lng = float(test_data['DEST_LNG'][i])
    cutoff_header.append(heading((CC_LAT, CC_LON),(cutoff_lat, cutoff_lng)))
    cutoff_distance_to_cc.append(calHarDist(cutoff_lat, cutoff_lng, CC_LAT, CC_LON))

test_data['HEADER_CUT_OFF_TO_CC'] = cutoff_header
test_data['CUT_OFF_DIST_FROM_CC'] = cutoff_distance_to_cc
test_data['CUT_OFF_LNG'] = test_data['DEST_LNG']
test_data['CUT_OFF_LAT'] = test_data['DEST_LAT']

print("Generating header from origin to cutoff")
# for train
origin_cutoff_header = []
origin_distance_to_cutoff = []

for i in range(modified_train.shape[0]):
    origin_lat = float(modified_train['ORIGIN_LAT'][i])
    origin_lng = float(modified_train['ORIGIN_LNG'][i])
    cutoff_lat = float(modified_train['CUT_OFF_LAT'][i])
    cutoff_lng = float(modified_train['CUT_OFF_LNG'][i])
    origin_cutoff_header.append(heading((origin_lat, origin_lng),(cutoff_lat, cutoff_lng)))
    origin_distance_to_cutoff.append(calHarDist(origin_lat, origin_lng, cutoff_lat, cutoff_lng))

modified_train['HEADER_ORIGIN_TO_CUT_OFF'] = origin_cutoff_header
modified_train['CUT_OFF_DIST_FROM_ORIGIN'] = origin_distance_to_cutoff


# for test
origin_cutoff_header = []
origin_distance_to_cutoff = []

for i in range(test_data.shape[0]):
    origin_lat = float(test_data['ORIGIN_LAT'][i])
    origin_lng = float(test_data['ORIGIN_LNG'][i])
    cutoff_lat = float(test_data['CUT_OFF_LAT'][i])
    cutoff_lng = float(test_data['CUT_OFF_LNG'][i])
    origin_cutoff_header.append(heading((origin_lat, origin_lng),(cutoff_lat, cutoff_lng)))
    origin_distance_to_cutoff.append(calHarDist(origin_lat, origin_lng, cutoff_lat, cutoff_lng))

test_data['HEADER_ORIGIN_TO_CUT_OFF'] = origin_cutoff_header
test_data['CUT_OFF_DIST_FROM_ORIGIN'] = origin_distance_to_cutoff



print("Merging Polyline")
#for train
poly_train = pd.DataFrame()
poly_train['TRIP_ID'] = train_data['TRIP_ID']
poly_train['POLYLINE'] = train_data['POLYLINE']
modified_train = pd.merge(modified_train, poly_train, on = 'TRIP_ID', how = 'inner')

print("Compute Cummulative distance, median velocity and final velocity")
#get harversine distance function
#for test

dist_list = []
median_velocity = []
final_velocity = []

for i in range(test_data.shape[0]):
    row = ast.literal_eval(test_data['POLYLINE'][i])
    cum_dist = 0
    temp = []
    temp_velocity = []
    for j in range(len(row)-1):
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

test_data['CUM_DIST'] = dist_list
test_data['MEDIAN_VELOCITY'] = median_velocity
test_data['FINAL_VELOCITY'] = final_velocity

#for train
dist_list = []
median_velocity = []
final_velocity = []
modified_train
for i in range(modified_train.shape[0]):
    row = ast.literal_eval(modified_train['POLYLINE'][i])
    cum_dist = 0
    temp = []
    temp_velocity = []
    for j in range(modified_train['CUT_OFF_LENGTH'][i] - 1):
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

modified_train['CUM_DIST'] = dist_list
modified_train['MEDIAN_VELOCITY'] = median_velocity
modified_train['FINAL_VELOCITY'] = final_velocity

print("Dropping redundant rows and parsing results")
# for test
modified_test = test_data.drop(columns = ['POLYLINE','DEST_LNG','DEST_LAT'])
modified_test = modified_test[['TRIP_ID', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DATE', 'END_TIME', 'DAY_OF_WEEK', 'HOUR', 'ORIGIN_LNG', 'ORIGIN_LAT', 'ORIGIN_HEADER', 'ORIGIN_DISTANCE_TO_CC', 'CUT_OFF_LENGTH', 'CUT_OFF_LAT', 'CUT_OFF_LNG', 'CUT_OFF_DIST_FROM_CC', 'HEADER_CUT_OFF_TO_CC', 'CUT_OFF_DIST_FROM_ORIGIN', 'HEADER_ORIGIN_TO_CUT_OFF', 'CUM_DIST', 'MEDIAN_VELOCITY', 'FINAL_VELOCITY', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'ACTUAL_DAYTYPE_A', 'ACTUAL_DAYTYPE_B', 'ACTUAL_DAYTYPE_C']]

# for train
modified_train = modified_train.drop(columns = ['POLYLINE','DEST_LNG','DEST_LAT'])
modified_train = modified_train[['TRIP_ID', 'ORIGIN_CALL', 'ORIGIN_STAND', 'TAXI_ID', 'TIMESTAMP', 'DATE', 'END_TIME', 'DAY_OF_WEEK', 'HOUR', 'ORIGIN_LNG', 'ORIGIN_LAT', 'ORIGIN_HEADER', 'ORIGIN_DISTANCE_TO_CC', 'CUT_OFF_LENGTH', 'CUT_OFF_LAT', 'CUT_OFF_LNG', 'CUT_OFF_DIST_FROM_CC', 'HEADER_CUT_OFF_TO_CC', 'CUT_OFF_DIST_FROM_ORIGIN', 'HEADER_ORIGIN_TO_CUT_OFF', 'CUM_DIST', 'MEDIAN_VELOCITY', 'FINAL_VELOCITY', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'ACTUAL_DAYTYPE_A', 'ACTUAL_DAYTYPE_B', 'ACTUAL_DAYTYPE_C', 'DURATION']]

print("Writing dataframe to CSV")

modified_train.to_csv('datasets/modified_train.csv', index=False)
modified_test.to_csv('datasets/modified_test.csv', index=False)

print("Completed in %s s" %(time.time()-starttime))
