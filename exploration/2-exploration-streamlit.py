import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import sys

sys.path.append('../')
from functions.graphs import gauge_chart


@st.cache_data
def load_tracking_data():
    return pd.read_csv('../data/tmp_tracking.csv')  #  load the tracking data
df_tracking = load_tracking_data()

# derived attributes
angles_knee = df_tracking[df_tracking['min_height_ankle'] == True]['angle_knee'].values
angles_elbow = df_tracking['angle_elbow'].values
angles_shoulder = df_tracking['angle_shoulder'].values


st.set_page_config(layout="wide")
st.title('Bike fitting 🚴')

st.markdown('###')

st.subheader('Summary') 

c1, c2 = st.columns(2)
c1.write('Knee angle')
c1.plotly_chart(gauge_chart(angles_knee, [120, 180], [140, 150], title='Knee angle - summary'))
c1.write('Elbow angle')
c1.plotly_chart(gauge_chart(angles_elbow, [90, 180], [155, 165], title='Elbow angle - summary'))
c1.write('Shoulder angle')
c1.plotly_chart(gauge_chart(angles_shoulder, [60, 150], [85, 95], title='Shoulder angle - summary'))

st.markdown('###')

st.subheader('Detailed analysis')

with st.expander('Detailed analysis - Knee angle'):
    st.subheader('Knee Angle') 
    c1_knee, c2_knee = st.columns(2)

    # select frames where the variable min_height is True in a dropdown
    frame = c1_knee.selectbox('Frames where the pedals are in a vertical position', df_tracking[df_tracking['min_height_ankle'] == True]['frame'].values)

    c1_knee.subheader('Distribution of knee angle')


    col11, col12, col13 = c1_knee.columns(3)
    col11.metric('Lower bound', int(np.min(angles_knee)))
    col12.metric('Average angle', int(np.mean(angles_knee)))
    col13.metric('Upper bound', int(np.max(angles_knee)))

    c1_knee.plotly_chart(gauge_chart(angles_knee, [120, 180], [140, 150]))

    frame_file = open(f'../data/tmp_frames/frame_{frame}.jpg', 'rb')
    frame_bytes = frame_file.read()
    c2_knee.image(frame_bytes)
