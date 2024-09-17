import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

# Load data from CSV file
try:
    data = pd.read_csv('path_to_your_csv_file.csv')
    print("CSV file loaded successfully.")
    print("Columns in the DataFrame:", data.columns.tolist())
    print("Shape of the DataFrame:", data.shape)
    print("\nFirst few rows of the data:")
    print(data.head())
except FileNotFoundError:
    print("Error: CSV file not found. Please check the file path.")
    exit()

# Check if 'label' column exists
if 'label' not in data.columns:
    print("\nError: 'label' column not found in the DataFrame.")
    print("Available columns:", data.columns.tolist())
    label_column = input("Please enter the name of the label column: ")
else:
    label_column = 'label'

# Separate features and labels
try:
    X = data.drop(label_column, axis=1).values
    y = data[label_column].values
    print("\nFeatures and labels separated successfully.")
    print("Shape of features (X):", X.shape)
    print("Shape of labels (y):", y.shape)
except KeyError as e:
    print(f"\nError: {e}")
    print("Failed to separate features and labels. Please check the column names.")
    exit()

# Create label mapping
unique_labels = np.unique(y)
label_map = {label: num for num, label in enumerate(unique_labels)}
print("\nUnique labels:", unique_labels)
print("Label mapping:", label_map)

# Convert labels to numeric
y_numeric = np.array([label_map[label] for label in y])

# Convert to categorical
y_categorical = to_categorical(y_numeric)

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15/90, random_state=2)

print("\nData split completed.")
print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")
