import pandas as pd
import numpy as np
import math

# Load the CSV data containing keypoints
df = pd.read_csv('normalized_keypoints.csv')

# 6.1: Calculate Angles Between Keypoints
def calculate_angle(x1, y1, x2, y2, x3, y3):
    """
    Calculate the angle between three points: (x1, y1), (x2, y2), and (x3, y3).
    This uses the atan2 function to calculate the angle.
    """
    angle = math.degrees(
        math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2)
    )
    return abs(angle) if angle >= 0 else 360 + angle

# Add angles to the dataframe
def add_angle_features(df):
    angles = []
    for i, row in df.iterrows():
        # Example: Calculate elbow angle between shoulder, elbow, and wrist
        left_elbow_angle = calculate_angle(row['left_shoulder_x'], row['left_shoulder_y'], 
                                           row['left_elbow_x'], row['left_elbow_y'], 
                                           row['left_wrist_x'], row['left_wrist_y'])
        
        right_elbow_angle = calculate_angle(row['right_shoulder_x'], row['right_shoulder_y'], 
                                            row['right_elbow_x'], row['right_elbow_y'], 
                                            row['right_wrist_x'], row['right_wrist_y'])
        
        # Example: Calculate knee angle between hip, knee, and ankle
        left_knee_angle = calculate_angle(row['left_hip_x'], row['left_hip_y'], 
                                          row['left_knee_x'], row['left_knee_y'], 
                                          row['left_ankle_x'], row['left_ankle_y'])
        
        right_knee_angle = calculate_angle(row['right_hip_x'], row['right_hip_y'], 
                                           row['right_knee_x'], row['right_knee_y'], 
                                           row['right_ankle_x'], row['right_ankle_y'])
        
        angles.append({
            'frame': row['frame'],
            'left_elbow_angle': left_elbow_angle,
            'right_elbow_angle': right_elbow_angle,
            'left_knee_angle': left_knee_angle,
            'right_knee_angle': right_knee_angle,
        })

    angles_df = pd.DataFrame(angles)
    return pd.concat([df, angles_df], axis=1)

# 6.2: Calculate Distances Between Keypoints
def calculate_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points (x1, y1) and (x2, y2).
    """
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
        
        # Example: Ankle-to-hip distance
        left_ankle_hip_distance = calculate_distance(row['left_ankle_x'], row['left_ankle_y'],
                                                     row['left_hip_x'], row['left_hip_y'])
        
        right_ankle_hip_distance = calculate_distance(row['right_ankle_x'], row['right_ankle_y'],
                                                      row['right_hip_x'], row['right_hip_y'])

        distances.append({
            'frame': row['frame'],
            'left_hand_shoulder_distance': left_hand_shoulder_distance,
            'right_hand_shoulder_distance': right_hand_shoulder_distance,
            'left_ankle_hip_distance': left_ankle_hip_distance,
            'right_ankle_hip_distance': right_ankle_hip_distance
        })

    distances_df = pd.DataFrame(distances)
    return pd.concat([df, distances_df], axis=1)

# Combine the new features and save the updated dataframe to CSV
def process_keypoints(df):
    # Add angles
    df = add_angle_features(df)
    
    # Add distances
    df = add_distance_features(df)
    
    # Save the processed dataframe to a new CSV file
    df.to_csv('processed_keypoint_features.csv', index=False)
    
    # Return the dataframe for immediate use
    return df

# Process the keypoints and save them into a new file
df = process_keypoints(df)

# Output the first few rows to verify
print(df.head())
