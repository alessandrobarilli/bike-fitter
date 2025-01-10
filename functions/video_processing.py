import cv2
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
from .tracking import update_landmarks_dict


# set random seed for reproducibility
np.random.seed(42)


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


def process_video(cap, path_video, frames_to_skip_beginning,
                  frames_to_skip_end, n_frames, fps, width, height, landmarks_dict,
                  create_output_video=False, out_filename='tmp.mp4'):
    """
    Process the video frame by frame and save the frames with the landmarks
    """

    # Define the codec and create VideoWriter object
    if create_output_video:
        fourcc = cv2.VideoWriter_fourcc(*'H264') # for streamlit compatibility
        out = cv2.VideoWriter(out_filename, fourcc, fps, (width,  height))

    frame_count = 0
    pose_undetected = 0
    frame_list = []

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    # is there a way to process it faster and not frame by frame?
    # possibility to save the edited video with landmarks and stuff
    print('Processing video...')
    while cap.isOpened():
        ret, frame = cap.read()
        frame_count += 1

        if not ret:
            break
        
        if  frames_to_skip_beginning <= frame_count < (n_frames - frames_to_skip_end):

            # Convert the frame to RGB: why this conversion to writeable?
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process the frame with MediaPipe Pose
            results = pose.process(image)
            
            # Convert the frame back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
            # Draw pose landmarks
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                update_landmarks_dict(results, width, height, landmarks_dict)
                frame_list.append(frame_count - frames_to_skip_beginning + 1)

                # Display angle close to the knee
                #cv2.rectangle(image, (int(knee[0]), int(knee[1]) - 20), (int(knee[0]) + 40, int(knee[1]) + 10), (0, 0, 0), -1)
                #cv2.putText(image, f'{int(angle)}', (int(knee[0]) + 5, int(knee[1]) + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (256, 256, 256), 1, cv2.LINE_AA)
            else:
                pose_undetected += 1

            if create_output_video:
                out.write(image)

            # save the image as compressed jpg 
            cv2.imwrite(f'data/tmp_frames/frame_{frame_count - frames_to_skip_beginning + 1}.jpg', image)
    print(f"Frames processed: {frame_count - frames_to_skip_beginning}")
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # without this, will not close properly
    cap.release()
    if create_output_video:
        out.release()
    
    return frame_list


def rescale_frame(frame, scale):

    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def find_scale_factor(frame, monitor_width, monitor_height):
    """
    Rescales the frame to the size of monitor
    """

    frame_width = frame.shape[1]
    frame_height = frame.shape[0]

    if (frame_width > monitor_width) or (frame_height > monitor_height):
        with_scale = monitor_width / frame_width
        height_scale = monitor_height / frame_height
        return min(with_scale, height_scale)
    
    if (frame_width < monitor_width) and (frame_height < monitor_height):
        with_scale = monitor_width / frame_width
        height_scale = monitor_height / frame_height
        return min(with_scale, height_scale)

    return 1


def get_coordinates_from_video(sample_imgs, monitor_width, monitor_height, msg='', type_coordinates='point'):
    
    coordinates_sampled = []
    
    for i, img in enumerate(sample_imgs):

        # does not depend on the monitor resolution
        scale = find_scale_factor(img, monitor_width, monitor_height)
        img_rescaled = rescale_frame(img, scale=scale)


        def capture_event(event, x, y, flags, param):
            captured = False

            # make a copy of the image
            img_rescaled_tmp = img_rescaled.copy()

            if event == cv2.EVENT_MOUSEMOVE:

                # plot a green vertical and horizontal line
                cv2.line(img_rescaled_tmp, (x, 0), (x, img_rescaled.shape[0]), (0, 255, 0), 1)
                cv2.line(img_rescaled_tmp, (0, y), (img_rescaled.shape[1], y), (0, 255, 0), 1)

                cv2.imshow("frame", img_rescaled_tmp)

            if event == cv2.EVENT_LBUTTONDOWN:
                if type_coordinates == 'point':
                    if msg:
                        print(msg)
                    else:
                        print(f"iteration {i + 1} of {len(sample_imgs)} - Captured coordinates ({x}, {y}), press any key to continue")
                    coordinates_sampled.append((x, y))
                    cv2.destroyWindow("frame")
                    
                if type_coordinates == 'line':
                    pass


        cv2.imshow('frame', img_rescaled)
        cv2.setMouseCallback('frame', capture_event)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    coordinates = ((1 / scale) * np.array(coordinates_sampled).mean(axis=0)).astype(int)
    print(f"Pedal coordinates obtained (original frame sizes): {coordinates}")
    return coordinates
