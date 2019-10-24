
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
           
DATA_DIR = '../datasets'


for filename in ['train_modified_v2.csv']:
    print('reading training data from %s ...' % filename)
    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    y = df['DURATION']
    df.drop(['Unnamed: 0', 'TRIP_ID', 'END_TIME', 'TIMESTAMP', 'DATE', 'DURATION', 'DEST_LNG', 'DEST_LAT'], axis=1, inplace=True)
    values = {'ORIGIN_CALL': -1, 'ORIGIN_STAND': -1}
    df = df.fillna(value=values)
    X = np.array(df, dtype=np.float)
    t0 = time.time()
    reg = LinearRegression().fit(X, y)
    print(reg.score(X, y))
    print('Done in %.1f sec.' % (time.time() - t0))
    
    df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))
    
    ids = df['TRIP_ID']
    df.drop(['Unnamed: 0', 'TRIP_ID', 'END_TIME', 'TIMESTAMP', 'DATE', 'CALL_TYPE_A', 'CALL_TYPE_B', 'CALL_TYPE_C', 'ACTUAL_DAYTYPE_A', 'ACTUAL_DAYTYPE_B', 'ACTUAL_DAYTYPE_C', 'DURATION', 'DEST_LNG', 'DEST_LAT'], axis=1, inplace=True)
    values = {'ORIGIN_CALL': -1, 'ORIGIN_STAND': -1}
    df = df.fillna(value=values)
    X_tst = np.array(df, dtype=np.float)
    y_pred = reg.predict(X_tst)
    
    submission = pd.DataFrame(ids, columns=['TRIP_ID'])
    submission['TRAVEL_TIME'] = y_pred
    submission.to_csv('../datasets/my_submission.csv', index=False)

    # print('training a random forest regressor ...')
    # # Initialize the famous Random Forest Regressor from scikit-learn
    # clf = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=21)
    # clf.fit(X, y)

    # print('predicting test data ...')
    # df = pd.read_csv(os.path.join(DATA_DIR, filename.replace('train', 'test')))
    # ids = df['TRIP_ID']
    
    # df = df.drop(['TRIP_ID', 'CALL_TYPE', 'TAXI_ID'], axis = 1)
    # X_tst = np.array(df, dtype=np.float)
    # y_pred = clf.predict(X_tst)

    # # create submission file
    # submission = pd.DataFrame(ids, columns=['TRIP_ID'])
    # filename = filename.replace('train_pp', 'my_submission')
    # submission['TRAVEL_TIME'] = np.exp(y_pred)
    # submission.to_csv(filename, index = False)

