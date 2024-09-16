import pathlib
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import einops

# Определение путей к данным
train_path = pathlib.Path("/home/olga/Pictures/Rehab-app/video_dataset/train")
val_path = pathlib.Path("/home/olga/Pictures/Rehab-app/video_dataset/val")
test_path = pathlib.Path("/home/olga/Pictures/Rehab-app/video_dataset/test")

class FrameGenerator:
    def __init__(self, path, n_frames, training=False):
        self.path = path
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(p.name for p in self.path.iterdir() if p.is_dir()))
        self.class_ids_for_name = dict((name, idx) for idx, name in enumerate(self.class_names))

    def get_files_and_class_names(self):
        video_paths = list(self.path.glob('*/*'))  # Предполагаем, что каждая папка содержит набор фреймов для одного видео
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes

    def __call__(self):
        video_paths, classes = self.get_files_and_class_names()
        pairs = list(zip(video_paths, classes))

        if self.training:
            random.shuffle(pairs)

        for path, name in pairs:
            frames = self.load_frames(path)
            label = self.class_ids_for_name[name]
            yield frames, label

    def load_frames(self, path):
        frame_files = sorted(path.glob('*.jpg'))[:self.n_frames]  # Предполагаем, что фреймы сохранены как JPG
        frames = []
        for frame_file in frame_files:
            frame = tf.io.read_file(str(frame_file))
            frame = tf.image.decode_jpeg(frame, channels=3)
            frame = tf.image.convert_image_dtype(frame, tf.float32)
            frames.append(frame)
        
        # Если фреймов меньше, чем n_frames, дополняем нулевыми фреймами
        if len(frames) < self.n_frames:
            padding = [tf.zeros_like(frames[0])] * (self.n_frames - len(frames))
            frames.extend(padding)
        
        return tf.stack(frames)

# Модельные классы и функции (оставляем без изменений)
class Conv2Plus1D(keras.layers.Layer):
    def __init__(self, filters, kernel_size, padding):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Conv3D(filters=filters, kernel_size=(1, kernel_size[1], kernel_size[2]), padding=padding),
            layers.Conv3D(filters=filters, kernel_size=(kernel_size[0], 1, 1), padding=padding)
        ])
    
    def call(self, x):
        return self.seq(x)

class ResidualMain(keras.layers.Layer):
    def __init__(self, filters, kernel_size):
        super().__init__()
        self.seq = keras.Sequential([
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization(),
            layers.ReLU(),
            Conv2Plus1D(filters=filters, kernel_size=kernel_size, padding='same'),
            layers.LayerNormalization()
        ])
    
    def call(self, x):
        return self.seq(x)

class Project(keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.seq = keras.Sequential([
            layers.Dense(units),
            layers.LayerNormalization()
        ])

    def call(self, x):
        return self.seq(x)

def add_residual_block(input, filters, kernel_size):
    out = ResidualMain(filters, kernel_size)(input)
    res = input
    if out.shape[-1] != input.shape[-1]:
        res = Project(out.shape[-1])(res)
    return layers.add([res, out])

# Параметры модели
n_frames = 30  # Предполагаем, что у нас 30 фреймов на видео
batch_size = 8
num_classes = 10  # Количество классов в вашем наборе данных
HEIGHT = 224  # Предполагаем, что все фреймы имеют размер 224x224
WIDTH = 224

# Создание наборов данных
output_signature = (tf.TensorSpec(shape=(n_frames, HEIGHT, WIDTH, 3), dtype=tf.float32),
                    tf.TensorSpec(shape=(), dtype=tf.int16))

train_ds = tf.data.Dataset.from_generator(FrameGenerator(train_path, n_frames, training=True),
                                          output_signature=output_signature)
train_ds = train_ds.batch(batch_size)

val_ds = tf.data.Dataset.from_generator(FrameGenerator(val_path, n_frames),
                                        output_signature=output_signature)
val_ds = val_ds.batch(batch_size)

test_ds = tf.data.Dataset.from_generator(FrameGenerator(test_path, n_frames),
                                         output_signature=output_signature)
test_ds = test_ds.batch(batch_size)

# Определение модели
input_shape = (n_frames, HEIGHT, WIDTH, 3)
input = layers.Input(shape=input_shape)
x = input

x = Conv2Plus1D(filters=16, kernel_size=(3, 7, 7), padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)

# Block 1
x = add_residual_block(x, 16, (3, 3, 3))

# Block 2
x = add_residual_block(x, 32, (3, 3, 3))

# Block 3
x = add_residual_block(x, 64, (3, 3, 3))

# Block 4
x = add_residual_block(x, 128, (3, 3, 3))

x = layers.GlobalAveragePooling3D()(x)
x = layers.Dense(num_classes)(x)

model = keras.Model(input, x)

# Компиляция модели
model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_ds, epochs=50, validation_data=val_ds)

# Функции для визуализации результатов
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.set_title('Loss')
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='val')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.set_title('Accuracy')
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='val')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# Оценка модели
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test accuracy: {test_accuracy:.4f}")

# Функции для анализа результатов
def get_actual_predicted_labels(dataset):
    actual = []
    predicted = []
    for x, y in dataset:
        actual.extend(y.numpy())
        predicted.extend(np.argmax(model.predict(x), axis=1))
    return np.array(actual), np.array(predicted)

def plot_confusion_matrix(actual, predicted, labels):
    cm = tf.math.confusion_matrix(actual, predicted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# Построение матрицы ошибок
fg = FrameGenerator(train_path, n_frames, training=True)
labels = list(fg.class_ids_for_name.keys())

actual, predicted = get_actual_predicted_labels(test_ds)
plot_confusion_matrix(actual, predicted, labels)

# Расчет метрик классификации
def calculate_classification_metrics(y_actual, y_pred, labels):
    cm = tf.math.confusion_matrix(y_actual, y_pred)
    tp = np.diag(cm)
    precision = dict()
    recall = dict()
    for i in range(len(labels)):
        col = cm[:, i]
        fp = np.sum(col) - tp[i]
        
        row = cm[i, :]
        fn = np.sum(row) - tp[i]
        
        precision[labels[i]] = tp[i] / (tp[i] + fp) if (tp[i] + fp) > 0 else 0
        recall[labels[i]] = tp[i] / (tp[i] + fn) if (tp[i] + fn) > 0 else 0
    
    return precision, recall

precision, recall = calculate_classification_metrics(actual, predicted, labels)
print("Precision:", precision)
print("Recall:", recall)