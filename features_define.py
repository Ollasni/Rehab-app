import pandas as pd
import numpy as np
import math

# Load the CSV data containing keypoints
df = pd.read_csv('normalized_keypoints.csv')

# 6.1: Calculate Angles Between Keypoints
def calculate_angle(x1, y1, x2, y2, x3, y3):
    # Calculate the angle between three points (p1: (x1, y1), p2: (x2, y2), p3: (x3, y3))
    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    )
    return abs(angle) if angle >= 0 else 360 + angle

# Add angles to the dataframe
def add_angle_features(df):
    angles = []
    for i, row in df.iterrows():
        # Example: Calculate elbow angle between shoulder, elbow, wrist
        left_elbow_angle = calculate_angle(row['left_shoulder_x'], row['left_shoulder_y'], 
                                           row['left_elbow_x'], row['left_elbow_y'], 
                                           row['left_wrist_x'], row['left_wrist_y'])
        
        right_elbow_angle = calculate_angle(row['right_shoulder_x'], row['right_shoulder_y'], 
                                            row['right_elbow_x'], row['right_elbow_y'], 
                                            row['right_wrist_x'], row['right_wrist_y'])

        angles.append({
            'frame': row['frame'],
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
        })

    angles_df = pd.DataFrame(angles)
    return pd.concat([df, angles_df], axis=1)

# 6.2: Calculate Distances Between Keypoints
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

# Add distances to the dataframe
def add_distance_features(df):
    distances = []
    for i, row in df.iterrows():
        # Example: Hand-to-shoulder distance for arm bends
        left_hand_shoulder_distance = calculate_distance(row['left_wrist_x'], row['left_wrist_y'],
                                                         row['left_shoulder_x'], row['left_shoulder_y'])
        
        right_hand_shoulder_distance = calculate_distance(row['right_wrist_x'], row['right_wrist_y'],
                                                          row['right_shoulder_x'], row['right_shoulder_y'])
        
        distances.append({
            'frame': row['frame'],
            'left_hand_shoulder_distance': left_hand_shoulder_distance,
            'right_hand_shoulder_distance': right_hand_shoulder_distance,
        })

    distances_df = pd.DataFrame(distances)
    return pd.concat([df, distances_df], axis=1)

# 6.3: Calculate Velocity of Keypoints
def add_velocity_features(df):
    velocities = []
    for i in range(1, len(df)):
        # Time between frames is assumed to be constant (use frame difference as proxy for time)
        time_interval = 1  # Assuming the time difference between consecutive frames is 1 unit (e.g., 1/30 second for 30 FPS)

        # Calculate velocity for left hand
        left_hand_velocity = calculate_distance(df.iloc[i]['left_wrist_x'], df.iloc[i]['left_wrist_y'],
                                                df.iloc[i-1]['left_wrist_x'], df.iloc[i-1]['left_wrist_y']) / time_interval
        
        # Calculate velocity for right hand
        right_hand_velocity = calculate_distance(df.iloc[i]['right_wrist_x'], df.iloc[i]['right_wrist_y'],
                                                 df.iloc[i-1]['right_wrist_x'], df.iloc[i-1]['right_wrist_y']) / time_interval
        
        velocities.append({
            'frame': df.iloc[i]['frame'],
            'left_hand_velocity': left_hand_velocity,
            'right_hand_velocity': right_hand_velocity,
        })

    velocities_df = pd.DataFrame(velocities)
    # Merge velocities_df with df, starting from the second frame (since the first frame can't have velocity)
    return pd.concat([df.iloc[1:].reset_index(drop=True), velocities_df], axis=1)

# 6.4: Combine Features for Training

# Add angle features (elbow angles)
df = add_angle_features(df)

# Add distance features (hand-to-shoulder distances)
df = add_distance_features(df)

# Add velocity features (movement of keypoints between frames)
df = add_velocity_features(df)

# Store the updated dataset for training
df.to_csv('keypoint_features_for_training.csv', index=False)

# Output the first few rows to verify
print(df.head())
