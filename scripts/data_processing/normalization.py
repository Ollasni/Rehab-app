import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv('/home/olga/Pictures/Rehab-app/all_videos_pose_keypoints.csv')

# Let's assume we have the following columns:
# 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y', etc.
# This will normalize the coordinates relative to the shoulder width

# 5.1: Normalization relative to the shoulder width
def normalize_keypoints(df):
    for i, row in df.iterrows():
        # Calculate the distance between shoulders as a normalization factor
        shoulder_width = np.sqrt((row['right_shoulder_x'] - row['left_shoulder_x'])**2 + 
                                 (row['right_shoulder_y'] - row['left_shoulder_y'])**2)
        
        if shoulder_width == 0:  # Avoid division by zero
            shoulder_width = 1
        
        # Normalize each keypoint by dividing by the shoulder width
        for key in df.columns:
            if '_x' in key or '_y' in key:  # Normalize both x and y coordinates
                df.at[i, key] = row[key] / shoulder_width

    return df

# Normalize the keypoints
df = normalize_keypoints(df)

# 5.2: Handling missing keypoints

# Option 1: Interpolate missing values
df.interpolate(method='linear', inplace=True)

# Option 2: Remove rows with missing values
# df.dropna(inplace=True)

# Save the updated dataframe to a new CSV file
df.to_csv('normalized_keypoints.csv', index=False)

# Output first few rows for verification
print(df.head())
