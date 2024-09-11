import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose

# Root directory containing subdirectories with video files
root_directory = '/home/olga/Pictures/VIDEO_scliced'

# Function to recursively list all .mp4 files in directories
def list_video_files(directory):
    video_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                video_files.append(os.path.join(root, file))
    return video_files

# Get all video files in the root directory and its subdirectories
video_files = list_video_files(root_directory)

# Data storage for keypoints from all videos
all_keypoints_data = []

# Process each video file
for video_file in video_files:
    cap = cv2.VideoCapture(video_file)
    
    keypoints_data = []  # Store keypoints for the current video

    with mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print(f"Finished processing {video_file}")
                break

            frame_count += 1

            # Convert the frame from BGR to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the image and extract pose landmarks
            results = pose.process(image)

            if results.pose_landmarks:
                # Extract required landmarks: shoulders, elbows, knees
                landmarks = results.pose_landmarks.landmark

                # Get x, y coordinates of required landmarks
                keypoints = {
                    'video_file': video_file,  # Add video file name for reference
                    'frame': frame_count,
                    'left_shoulder_x': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                    'left_shoulder_y': landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y,
                    'right_shoulder_x': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                    'right_shoulder_y': landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y,
                    'left_elbow_x': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                    'left_elbow_y': landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y,
                    'right_elbow_x': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                    'right_elbow_y': landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y,
                    'left_knee_x': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                    'left_knee_y': landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y,
                    'right_knee_x': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                    'right_knee_y': landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y,
                }

                # Append keypoints for this frame to the list
                keypoints_data.append(keypoints)

        # Store the keypoints from the current video into the overall list
        all_keypoints_data.extend(keypoints_data)

    # Release the video capture for the current video
    cap.release()

# Convert the accumulated keypoints data to a pandas DataFrame
df = pd.DataFrame(all_keypoints_data)

# Save the DataFrame to a CSV file
df.to_csv('all_videos_pose_keypoints.csv', index=False)

# Optionally, output the DataFrame to inspect
print(df.head())
