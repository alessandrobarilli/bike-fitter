import streamlit as st
import pandas as pd
import numpy as np
import sys

from functions.graphs import gauge_chart

st.set_page_config(layout="wide")

@st.cache_data
def load_tracking_data():
    return pd.read_csv('data/tmp_tracking.csv')  #  load the tracking data
df_tracking = load_tracking_data()

# derived attributes
angles_knee = df_tracking[df_tracking['min_height_ankle'] == True]['angle_knee'].values
angles_elbow = df_tracking['angle_elbow'].values
angles_shoulder = df_tracking['angle_shoulder'].values
angles_torso = df_tracking['angle_torso'].values

interactive = False
if 'knee_over_pedal' in df_tracking.columns:
    interactive = True
    knee_over_pedal = df_tracking['knee_over_pedal'].values[0]

st.title('Bike fitting 🚴')

st.markdown('###')

st.subheader('Summary') 

c1, c2 = st.columns(2)
if interactive:
    c1.write('Knee over pedal distance: optimal range [-3, 1] (cm)')
    c1.plotly_chart(gauge_chart(knee_over_pedal, [-10, 10], [-3, 1], title='Knee over pedal distance - summary'))
    c1.markdown('######')
c1.write('Knee angle: optimal range [140, 150]')
c1.plotly_chart(gauge_chart(angles_knee, [120, 180], [140, 150], title='Knee angle - summary'))
c1.markdown('######')
c1.write('Elbow angle: optimal range [155, 165]')
c1.plotly_chart(gauge_chart(angles_elbow, [90, 180], [155, 165], title='Elbow angle - summary'))

c2.write('Shoulder angle: optimal range [85, 95]')
c2.plotly_chart(gauge_chart(angles_shoulder, [60, 150], [85, 95], title='Shoulder angle - summary'))
c2.markdown('######')
c2.write('Torso angle: optimal range [30, 45]')
c2.plotly_chart(gauge_chart(angles_torso, [20, 80], [30, 45], title='Torso angle - summary'))

st.markdown('###')
st.subheader('Detailed analysis')


def detailed_section(title, val, possible_range, recommended_range, dropdown_msg, type_value='angle', type_pedals=None):


    with st.expander(f'Detailed analysis: {title}'):
        st.subheader(title) 
        c1_knee, c2_knee = st.columns(2)

        if type_pedals == 'vertical':
            frame = c1_knee.selectbox(dropdown_msg, df_tracking[df_tracking['min_height_ankle'] == True]['frame'].values, key=f'{title}_dropdown')
        else:
            frame = c1_knee.selectbox('Select a frame', df_tracking['frame'].values, key=f'{title}_dropdown')
        

        c1_knee.subheader(f'Distribution of {title.lower()}')

        col11, col12, col13 = c1_knee.columns(3)
        col11.metric(f'Minimum {type_value} observed', int(np.min(val)))
        col12.metric(f'Average {type_value} observed', int(np.mean(val)))
        col13.metric(f'Maximum {type_value} observed', int(np.max(val)))

        c1_knee.subheader(f'Recommended ranges of {title.lower()}')

        col11, col12 = c1_knee.columns(2)
        col11.metric(f'Minimum {type_value} recommended', recommended_range[0])
        col12.metric(f'Maximum {type_value} recommended', recommended_range[1])

        c1_knee.plotly_chart(gauge_chart(val, possible_range, recommended_range))

        frame_file = open(f'data/tmp_frames/frame_{frame}.jpg', 'rb')
        frame_bytes = frame_file.read()
        c2_knee.image(frame_bytes)


detailed_section(title='Knee angle', possible_range=[120, 180], recommended_range=[140, 150],
                 val=angles_knee, dropdown_msg='Frames where the pedals are in a vertical position', type_pedals='vertical')
detailed_section(title='Elbow angle', possible_range=[90, 180], recommended_range=[155, 165], val=angles_elbow, dropdown_msg='All frames')
detailed_section(title='Shoulder angle', possible_range=[60, 150], recommended_range=[85, 95], val=angles_shoulder, dropdown_msg='All frames')
detailed_section(title='Torso angle', possible_range=[20, 80], recommended_range=[30, 45], val=angles_torso, dropdown_msg='All frames')

with st.expander('All frames'):
    all_frames = st.selectbox('Select a frame', df_tracking['frame'].values)
    frame_file = open(f'data/tmp_frames/frame_{all_frames}.jpg', 'rb')
    frame_bytes = frame_file.read()
    st.image(frame_bytes)