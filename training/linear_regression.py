import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import sys
sys.path.append('..')
from utils import calHarDist
           
DATA_DIR = '../datasets'

for filename in ['modified_train_v3.csv']:
    print('reading training data from %s ...' % filename)
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df.drop(['TRIP_ID'], axis=1, inplace=True)
    c = df.corr().abs()
    print(c)
    y = df['DURATION']
<<<<<<< HEAD
    # df['displacement'] = df.apply(lambda row : calHarDist(row['ORIGIN_LAT'], row['ORIGIN_LNG'], row['CUT_OFF_LAT'], row['CUT_OFF_LNG']), axis = 1)
    df.drop(['TRIP_ID','TIMESTAMP','DATE', 'END_TIME', 'ORIGIN_CALL', 'ORIGIN_STAND', 'DURATION'], axis=1, inplace=True)
    # values = {'ORIGIN_CALL': -1, 'ORIGIN_STAND': -1}
    # df = df.fillna(value=values)
    
    
    X = np.array(df, dtype=np.float)
    # th1 = np.percentile(df['displacement'], [99.9])[0]
    # relevant_rows = (df['displacement'] < th1)
    # df.drop(['displacement'], axis=1, inplace=True)
    # X = df.loc[relevant_rows]
    # y = y.loc[relevant_rows]
=======
    df['displacement'] = df.apply(lambda row : calHarDist(row['ORIGIN_LAT'], row['ORIGIN_LNG'], row['DEST_LAT'], row['DEST_LNG']), axis = 1)
    df.drop(['END_TIME', 'TIMESTAMP', 'DATE', 'DURATION', 'DEST_LNG', 'DEST_LAT'], axis=1, inplace=True)
    values = {'ORIGIN_CALL': -1, 'ORIGIN_STAND': -1}
    df = df.fillna(value=values)
    
    
    X = np.array(df, dtype=np.float)
    th1 = np.percentile(df['displacement'], [99.9])[0]
    relevant_rows = (df['displacement'] < th1)
    # df.drop(['displacement'], axis=1, inplace=True)
    X = df.loc[relevant_rows]
    y = np.log(y.loc[relevant_rows])
>>>>>>> b0c57ef4d1b28a5c4bdd9ef435265bc3a129a135
    t0 = time.time()
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    print('Done in %.1f sec.' % (time.time() - t0))
    
    
    
    df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))

    ids = df['TRIP_ID']
    df.drop(['TRIP_ID','TIMESTAMP','DATE', 'END_TIME', 'ORIGIN_CALL', 'ORIGIN_STAND'], axis=1, inplace=True)
    # values = {'ORIGIN_CALL': -1, 'ORIGIN_STAND': -1}
    # df = df.fillna(value=values)
    X_tst = np.array(df, dtype=np.float)
    y_pred = reg.predict(X_tst)
    
    submission = pd.DataFrame(ids, columns=['TRIP_ID'])
    submission['TRAVEL_TIME'] = y_pred
    submission.to_csv('../datasets/my_submission.csv', index=False)

