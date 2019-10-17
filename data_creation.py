import numpy as np
import pandas as pd
from datetime import datetime
import pytz

local_tz = pytz.timezone('Portugal')
test_data = pd.read_csv('datasets/test.csv')
print(test_data.head)

train_data = pd.read_csv('datasets/train with actual day v3.csv')
# train_data.head()
train_data.groupby('MISSING_DATA').count()

holiday_data = pd.read_csv('datasets/Holidays.csv')
holiday_data.head()

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

#for test data
import ast
import matplotlib.pyplot as plt

counter = 0
count_md = test_data['MISSING_DATA'].count()
trip_duration = []
end_time = []
origin = []
dest = []
for i in range(count_md):
    polyline = ast.literal_eval(test_data['POLYLINE'][i])
    dur = len(polyline)
    trip_duration.append(dur)
    end_epoch_datestamp = test_data['TIMESTAMP'][i] + dur 
    end_datetime_for_datestamp_in_utc = datetime.utcfromtimestamp(end_epoch_datestamp)
    actual_end_datetime_for_datestamp = end_datetime_for_datestamp_in_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
    end_time.append(actual_end_datetime_for_datestamp)
    poly_arr = polyline
    if(poly_arr==[]):
        origin.append("NULL")
        dest.append("NULL")
    else:
        origin.append(poly_arr[0])
        dest.append(poly_arr[-1])
    counter+=1
    if(counter%500==0):
        print(counter/1710670)
test_data['DURATION'] = trip_duration
test_data['END_TIME'] = end_time
test_data['ORIGIN'] = origin
test_data['DESTINATION'] = dest

#for train data
counter = 0
count_md = train_data['MISSING_DATA'].count()
trip_duration = []
end_time = []
origin = []
dest = []
for i in range(count_md):
    try:
        polyline = ast.literal_eval(train_data['POLYLINE'][i])
        dur = len(polyline)
        trip_duration.append(dur)
        end_epoch_datestamp = train_data['TIMESTAMP'][i] + dur 
        end_datetime_for_datestamp_in_utc = datetime.utcfromtimestamp(end_epoch_datestamp)
        actual_end_datetime_for_datestamp = end_datetime_for_datestamp_in_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
        end_time.append(actual_end_datetime_for_datestamp)
        poly_arr = polyline
        if(poly_arr==[]):
            origin.append("NULL")
            dest.append("NULL")
        else:
            origin.append(poly_arr[0])
            dest.append(poly_arr[-1])
    except:
        trip_duration.append("NULL")
        end_time.append("NULL")
        origin.append("NULL")
        dest.append("NULL")
    counter+=1
    if(counter%500==0):
        print(counter/1710670)
train_data['DURATION'] = trip_duration
train_data['END_TIME'] = end_time
train_data['ORIGIN'] = origin
train_data['DESTINATION'] = dest

for index, row in train_data.iterrows():
    if(row['MISSING_DATA']== True):
        train_data.drop(index, inplace = True)
    
train_data.to_csv('datasets/train with duration origin etc v4.csv')

for index, row in train_data.iterrows():
    if(row['DURATION']== "NULL"):
        train_data.drop(index, inplace = True)
