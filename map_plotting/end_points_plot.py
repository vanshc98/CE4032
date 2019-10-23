import folium
import pandas as pd

m = folium.Map(
    location=[41.15767687592546, -8.615393063941816],
    zoom_start=12
)

train_data = pd.read_csv('../datasets/train_interim_6.csv')

for i in range(train_data.shape[0]):
    lat = float(train_data['DEST_LAT'][i])
    lon = float(train_data['DEST_LNG'][i])
    folium.Marker([lat, lon]).add_to(m)

m.save('index.html')


