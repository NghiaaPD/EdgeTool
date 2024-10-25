import streamlit as st
import pandas as pd
import random
import pydeck as pdk

def Map(data_ll):
    data_ll.columns = data_ll.columns.str.lower()

    if 'latitude' in data_ll.columns and 'longitude' in data_ll.columns:
        lat_min, lat_max = min(data_ll.latitude), max(data_ll.latitude)
        lon_min, lon_max = min(data_ll.longitude), max(data_ll.longitude)

        random_lat = random.uniform(lat_min, lat_max)
        random_lon = random.uniform(lon_min, lon_max)

        random_point_df = pd.DataFrame({
            "latitude": [random_lat],
            "longitude": [random_lon],
            "color": [[0, 255, 0]]
        })

        data_ll['color'] = [[255, 0, 0]] * len(data_ll)

        combined_df = pd.concat([data_ll[['latitude', 'longitude', 'color']], random_point_df])

        layer = pdk.Layer(
            'ScatterplotLayer',
            data=combined_df,
            get_position='[longitude, latitude]',
            get_color='color',
            get_radius=200,
        )

        view_state = pdk.ViewState(
            latitude=combined_df['latitude'].mean(),
            longitude=combined_df['longitude'].mean(),
            zoom=10,
            pitch=0,
        )

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
        )

        st.pydeck_chart(r)
        return random_lat, random_lon
    else:
        st.write("Latitude and Longitude columns not found in the DataFrame")
