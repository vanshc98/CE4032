import folium
from folium import plugins
import pandas as pd

from folium.plugins import FastMarkerCluster

train_data = pd.read_csv('../datasets/train_forvansh.csv')

dummy2 = pd.DataFrame(train_data, columns=['ORIGIN_LAT', 'ORIGIN_LNG'])
dummy2 = dummy2.values  ## convert to Numpy array
d2map = folium.Map(location = [41.15767687592546, -8.615393063941816], zoom_start = 12)
plugins.FastMarkerCluster(dummy2).add_to(d2map)
d2map.save("MarkerPlot.html") 

m = folium.Map(
    location=[41.15767687592546, -8.615393063941816],
    zoom_start=12
)

stationArr = train_data[['DEST_LAT', 'DEST_LNG']].as_matrix()

m.add_child(plugins.HeatMap(stationArr, radius=15))
m.save('DestHeatMap.html')

m2 = folium.Map(
    location=[41.15767687592546, -8.615393063941816],
    zoom_start=12
)

stationArr = train_data[['ORIGIN_LAT', 'ORIGIN_LNG']].as_matrix()

m2.add_child(plugins.HeatMap(stationArr, radius=15))
m2.save('OriginHeatMap.html')



