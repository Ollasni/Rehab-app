import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

# Load data from CSV file
# Adjust the path and column names as per your CSV file structure
data = pd.read_csv('/home/olga/Pictures/Rehab-app/keypoint_features_for_training.csv')

# Assuming your CSV has columns for features and a 'label' column
X = data.drop('label', axis=1).values
y = data['label'].values

# Create label mapping
unique_labels = np.unique(y)
label_map = {label: num for num, label in enumerate(unique_labels)}

# Convert labels to numeric
y_numeric = np.array([label_map[label] for label in y])

# Convert to categorical
y_categorical = to_categorical(y_numeric)

# Reshape X if necessary (e.g., for LSTM, you might need to reshape to 3D)
# X = X.reshape(X.shape[0], timesteps, features)

# Split data into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.1, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15/90, random_state=2)

print(f"Training set shape: {X_train.shape}, {y_train.shape}")
print(f"Validation set shape: {X_val.shape}, {y_val.shape}")
print(f"Test set shape: {X_test.shape}, {y_test.shape}")

# Callbacks to be used during neural network training 
es_callback = EarlyStopping(monitor='val_loss', min_delta=5e-4, patience=10, verbose=0, mode='min')
lr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=0, mode='min')
chkpt_callback = ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', verbose=0, save_best_only=True, 
                                 save_weights_only=False, mode='min', save_freq='epoch')

# Optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

# Some hyperparameters
batch_size = 32
max_epochs = 500