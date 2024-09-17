from flash.core.classification import ClassificationAdapterTask
from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional, Union
import torch
import os
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DistributedSampler
from torchmetrics import Accuracy
import flash
from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.providers import _PYTORCHVIDEO
from flash.core.utilities.types import (
    LOSS_FN_TYPE,
    LR_SCHEDULER_TYPE,
    METRICS_TYPE,
    OPTIMIZER_TYPE,
)
from utils import *

def get_files(directory, extensions=['.mp4']):
    """
    Рекурсивно получает все файлы с указанными расширениями из заданной директории.
    
    :param directory: Путь к директории для поиска
    :param extensions: Список расширений файлов для поиска
    :return: Список путей к файлам
    """
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(Path(os.path.join(root, filename)))
    return files

exercises = [
    "HEEL WALKING",
    "PINCH GRIP",
    "PLANK",
    "REVERSE LUNGE",
    "SCAPULA PROTRACTION",
    "TREE",
    "TRICEP DIPS",
    "TRICEP EXTENSION",
    "WAITERS BOW",
    "Y BALANCE"
]

def one_hot_encode(exercise, exercises):
    return [1 if ex == exercise else 0 for ex in exercises]

def multiBinary(vids):
    l = []
    for vid in vids:
        exercise = vid.parent.name  # Assuming the parent folder name is the exercise name
        lab = one_hot_encode(exercise, exercises)
        l.append(lab)
    return np.array(l)

def create_dataframe(data_path):
    data_path = Path(data_path)
    vids = get_files(data_path, extensions=[".mp4"])
    l = multiBinary(vids)
    vids = [str(vid).replace(str(data_path) + "/", "") for vid in vids]

    df = pd.DataFrame(
        {
            "video": vids,
            "HEEL WALKING": l[:, 0],
            "PINCH GRIP": l[:, 1],
            "PLANK": l[:, 2],
            "REVERSE LUNGE": l[:, 3],
            "SCAPULA PROTRACTION": l[:, 4],
            "TREE": l[:, 5],
            "TRICEP DIPS": l[:, 6],
            "TRICEP EXTENSION": l[:, 7],
            "WAITERS BOW": l[:, 8],
            "Y BALANCE": l[:, 9]
        }
    )

    # Перемешиваем DataFrame
    df = df.sample(frac=1, random_state=42)

    # Сохраняем DataFrame в CSV файл
    csv_filename = f"{data_path.name}_data.csv"
    df.to_csv(csv_filename, index=False)
    print(f"DataFrame сохранен в {csv_filename}")

    return df

# Обработка тренировочных данных
train_data_path = "/home/olga/Pictures/Rehab-app/video_dataset/train"
train_df = create_dataframe(train_data_path)

# Вывод первых нескольких строк тренировочных данных для проверки
print("Тренировочные данные:")
print(train_df.head())

# Обработка валидационных данных
val_data_path = "/home/olga/Pictures/Rehab-app/video_dataset/val"
val_df = create_dataframe(val_data_path)

# Вывод первых нескольких строк валидационных данных для проверки
print("\nВалидационные данные:")
print(val_df.head())