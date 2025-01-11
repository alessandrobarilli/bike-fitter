import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import streamlit as st
import argparse

from functions.geometry import angle_between, euclidean_distance
from functions.tracking import landmarks_dict
from functions.video_processing import process_video, get_coordinates_from_video
from functions.graphs import gauge_chart


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
tube_length_cm = 54
seconds_to_skip_beginning = 30
seconds_to_skip_end = 30
path_video = 'data/IMG_1476.mov'
monitor_width = 3440
monitor_height = 1440
image_samples = 2

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

def preprocess_data(interactive_mode):
    """
    Main function to process the video and calculate angles and distances.
    If interactive_mode is True, the user will be prompted to select the pedal and knee coordinates.
    Otherwise, the other coordinates will be calculated automatically.
    """

    print(f'\nProcess launched with interactive mode: {interactive_mode}')

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

    # if true, opens the video window to allow user to select the coordinates
    if interactive_mode:

        # calculating knee over pedal distance
        pedal_coordinates_sampled = []
        tube_coordinates_sampled = []
        frames_parallel_pedals = df_tracking[df_tracking['parallel_pedals'] == True]

        sample_frames = frames_parallel_pedals.sample(image_samples)['frame'].values
        parallel_pedal_images = []
        for i, sample_frame in enumerate(sample_frames):
            parallel_pedal_images.append(cv2.imread(f'data/tmp_frames/frame_{sample_frame}.jpg'))

        pedal_coordinates_sampled = get_coordinates_from_video(parallel_pedal_images, monitor_width, monitor_height, msg='', type_coordinates='point')
        knee_coordinates_sampled = get_coordinates_from_video(parallel_pedal_images, monitor_width, monitor_height, msg='', type_coordinates='point')
        tube_coordinates_sampled = get_coordinates_from_video([parallel_pedal_images[0]], monitor_width, monitor_height, msg='', type_coordinates='line')
        knee_pedal_distance_pixels = abs(pedal_coordinates_sampled.mean(axis=0).astype(int)[0]- knee_coordinates_sampled.mean(axis=0).astype(int)[0])
        tube_length_pixels = euclidean_distance(tube_coordinates_sampled[0], tube_coordinates_sampled[1])

        print(f'pedal estimated position on img: {pedal_coordinates_sampled.mean(axis=0).astype(int)}')
        print(f'knee estimated position on img: {knee_coordinates_sampled.mean(axis=0).astype(int)}')
        print(f'tube length in pixels: {tube_length_pixels}')
        print(f'knee pedal distance in pixels: {knee_pedal_distance_pixels}')
        print(f'knee pedal distance in cm: {(knee_pedal_distance_pixels / tube_length_pixels) * tube_length_cm}')

        df_tracking['knee_over_pedal'] = (knee_pedal_distance_pixels / tube_length_pixels) * tube_length_cm

    df_tracking.to_csv('data/tmp_tracking.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video with optional interactive mode.")
    parser.add_argument('--interactive', type=str, default='', help='Enable interactive mode (default: False)')

    args = parser.parse_args()

    if args.interactive == 'True':
        preprocess_data(True)
    elif args.interactive == 'False':
        preprocess_data(False)
    else:
        print('\nERROR: Please provide a valid argument for interactive mode: True or False\n')
