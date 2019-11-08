import math
import pandas as pd
import numpy as np

CC_LON = -8.615393063941816
CC_LAT = 41.15767687592546

def calHarDist(lat1, lon1, lat2, lon2):  # generally used geo measurement function
    R = 6378.137 # Radius of earth in KM
    dLat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dLon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dLat/2) * math.sin(dLat/2) +\
    math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) *\
    math.sin(dLon/2) * math.sin(dLon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = R * c
    return d * 1000 # meters

def one_hot(df, column_type):
    if column_type == 0:
        column = 'CALL_TYPE'
    elif column_type == 1:
        column = 'ACTUAL_DAYTYPE'
    one_hot = pd.get_dummies(df[column])
    one_hot = one_hot.rename(columns={"A": column + "_A", "B": column + "_B", "C":column + "_C"})
    return one_hot

def heading(p1, p2):
    p1 = np.radians(p1)
    p2 = np.radians(p2)
    lat1, lon1 = p1[0], p1[1]
    lat2, lon2 = p2[0], p2[1]
    aa = np.sin(lon2 - lon1) * np.cos(lat2)
    bb = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    return np.arctan2(aa, bb)
