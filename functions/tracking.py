import numpy as np


landmarks_mapping = {
    'left_hip': 23,
    'left_knee': 25,
    'left_ankle': 27,
    'right_ankle': 28,
    'left_shoulder': 11,
    'left_elbow': 13,
    'left_wrist': 15,
}


def update_landmarks_dict(
        results,
        width: int,
        height: int,
        landmarks_dict: dict
) -> dict:
    
    # Extract all landmarks results
    landmarks = results.pose_landmarks.landmark
    
    for landmark, landmark_map in landmarks_mapping.items():
        landmark_value_list = [
            width*landmarks[landmark_map].x,
            height*landmarks[landmark_map].y
        ]
        landmarks_dict[landmark].append(np.array(landmark_value_list))

    return landmarks_dict
