import cv2
import mediapipe as mp
import pandas as pd
import os

from functions.geometry import angle_between
from functions.tracking import landmarks_dict
from functions.video_processing import process_video, get_coordinates_from_video

for file in os.listdir('data/tmp_frames'):
    os.remove(f'data/tmp_frames/{file}')

create_output_video = False
local_minimum_threshold = 0.005
degree_threshold_parallel_pedals = 3

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# inputs
seconds_to_skip_beginning = 30
seconds_to_skip_end = 15
path_video = 'data/IMG_1476.mov'
monitor_width = 3440
monitor_height = 1440

# output (optional)
out_filename = 'tmp.mp4'

# variables
cap = cv2.VideoCapture(path_video)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration_seconds = n_frames/fps
frames_to_skip_beginning = int(round(seconds_to_skip_beginning * fps))
frames_to_skip_end = int(round(seconds_to_skip_end * fps))

frame_list = process_video(cap, path_video, frames_to_skip_beginning,
                          frames_to_skip_end, n_frames, fps,
                          width, height, landmarks_dict)

df_tracking = pd.DataFrame({
    'frame': frame_list,
    'hip': landmarks_dict['left_hip'],
    'knee': landmarks_dict['left_knee'],
    'ankle': landmarks_dict['left_ankle'],
    'ankle_right': landmarks_dict['right_ankle'],
    'shoulder': landmarks_dict['left_shoulder'],
    'elbow': landmarks_dict['left_elbow'],
    'wrist': landmarks_dict['left_wrist'],
})

# calculate the angle between the hip, knee and ankle
df_tracking['angle_knee'] = df_tracking.apply(lambda x: angle_between(x['knee'], x['hip'], x['ankle']), axis=1)
df_tracking['angle_shoulder'] = df_tracking.apply(lambda x: angle_between(x['shoulder'], x['hip'], x['elbow']), axis=1)
df_tracking['angle_elbow'] = df_tracking.apply(lambda x: angle_between(x['elbow'], x['shoulder'], x['wrist']), axis=1)
df_tracking['angle_torso'] = df_tracking.apply(lambda x: angle_between(x['hip'], x['shoulder'], x['hip'] - (1, 0)), axis=1)
df_tracking['angle_pedal'] = df_tracking.apply(lambda x: angle_between(x['ankle_right'], x['ankle'], x['ankle_right'] - (1, 0)), axis=1)

# boolean columns to identify positions
min_ankle = max([ankle[1] for ankle in df_tracking['ankle']])
df_tracking['min_height_ankle'] = df_tracking['ankle'].apply(lambda x: abs(x[1] - min_ankle)/min_ankle < local_minimum_threshold)
df_tracking['parallel_pedals'] = df_tracking['angle_pedal'].apply(lambda x: abs(x) < degree_threshold_parallel_pedals)

# calculating knee over pedal distance
pedal_coordinates_sampled = []
tube_coordinates_sampled = []
frames_parallel_pedals = df_tracking[df_tracking['parallel_pedals'] == True]

sample_frames = frames_parallel_pedals.sample(3)['frame'].values
parallel_pedal_images = []
for i, sample_frame in enumerate(sample_frames):
    parallel_pedal_images.append(cv2.imread(f'data/tmp_frames/frame_{sample_frame}.jpg'))

pedal_coordinates_sampled = get_coordinates_from_video(parallel_pedal_images, monitor_width, monitor_height, msg='', type_coordinates='point')


print(pedal_coordinates_sampled)





df_tracking.to_csv('data/tmp_tracking.csv', index=False)