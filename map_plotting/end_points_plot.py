import folium
import pandas as pd

import folium
from folium import plugins
import pandas as pd

m = folium.Map(
    location=[41.15767687592546, -8.615393063941816],
    zoom_start=12
)

train_data = pd.read_csv('../datasets/train_forvansh.csv')

stationArr = train_data[['DEST_LAT', 'DEST_LNG']].as_matrix()

m.add_child(plugins.HeatMap(stationArr, radius=15))
m.save('index.html')

m2 = folium.Map(
    location=[41.15767687592546, -8.615393063941816],
    zoom_start=12
)

stationArr = train_data[['ORIGIN_LAT', 'ORIGIN_LNG']].as_matrix()

m.add_child(plugins.HeatMap(stationArr, radius=15))
m.save('index2.html')

# for i in tqdm(range(train_data.shape[0])):
#     lat = float(train_data['DEST_LAT'][i])
#     lon = float(train_data['DEST_LNG'][i])
#     folium.Marker([lat, lon]).add_to(m)

# m.save('index.html')


# some_map = folium.Map(location=[41.15767687592546, -8.615393063941816], 
#  zoom_start=12)
# mc = MarkerCluster()
# #creating a Marker for each point in df_sample. Each point will get a popup with their zip
# for row in tqdm(range(train_data.shape[0])):
#     lat = float(train_data['DEST_LAT'][row])
#     lon = float(train_data['DEST_LNG'][row])
#     mc.add_child(folium.Marker(location=[lat,  lon]))
 
# some_map.add_child(mc)
# some_map.save('index.html')



