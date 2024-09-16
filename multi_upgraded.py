from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional, Union, Callable
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="flash")

import torch
import os
import pandas as pd
import numpy as np
from pathlib import Path
from torch import nn, Tensor
import torchvision.transforms.functional as F
from torch.utils.data import DistributedSampler
from flash.video import VideoClassificationData
from torchmetrics import Accuracy
import kornia as K
import flash
from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.data.io.input_transform import InputTransform
from flash.core.data.transforms import ApplyToKeys
from pytorchvideo.transforms import UniformTemporalSubsample 
from torchvision.transforms.functional import normalize
from lightning_utilities.core.apply_func import apply_to_collection

import torch
import torchvision.transforms as T

from flash.core.utilities.types import (
    LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE

)
from torchvision.transforms import RandomCrop, CenterCrop, Compose

from utils import *  # Assuming you have custom utility functions here

print(hasattr(K, 'VideoSequential'))

# Utility to fetch video files
def get_files(directory, extensions=['.mp4']):
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                files.append(Path(os.path.join(root, filename)))
    return files

# Function for one-hot encoding labels
def one_hot_encode(exercise, exercises):
    return [1 if ex == exercise else 0 for ex in exercises]

# DataFrame creation and CSV generation
def createMultiLabelDf(data_path):
    data_path = Path(data_path)
    vids = get_files(data_path, extensions=[".mp4"])
    l = multiBinary(vids)  # Assuming you have this utility
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
            "Y BALANCE": l[:, 9],
        }
    )
    df.to_csv(f"{data_path}/{data_path.name}.csv", index=False)

def normalize_tensor(tensor: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    return (tensor - mean[None, :, None, None]) / std[None, :, None, None]

# Update the per_batch_transform_on_device method
def per_batch_transform_on_device(self) -> Callable:
    return ApplyToKeys(
        DataKeys.INPUT,
        Compose([
            T.Lambda(lambda x: normalize_tensor(x, self.mean, self.std)),
            # Add other transformations if needed
        ]),
    )

# Transform class to handle input transformations
class VideoClassificationInputTransform(InputTransform):
    image_size: int = 244
    temporal_sub_sample: int = 8
    mean: Tensor = torch.tensor([0.45, 0.45, 0.45])
    std: Tensor = torch.tensor([0.225, 0.225, 0.225])
    data_format: str = "BCTHW"
    same_on_frame: bool = False

    def per_sample_transform(self) -> Callable:
        per_sample_transform = [CenterCrop(self.image_size)]
        return ApplyToKeys(
            DataKeys.INPUT,
            Compose([UniformTemporalSubsample(self.temporal_sub_sample), normalize] + per_sample_transform),
        )

    def train_per_sample_transform(self) -> Callable:
        per_sample_transform = [RandomCrop(self.image_size, pad_if_needed=True)]
        return ApplyToKeys(
            DataKeys.INPUT,
            Compose([UniformTemporalSubsample(self.temporal_sub_sample), normalize] + per_sample_transform),
        )

    def per_batch_transform_on_device(self) -> Callable:
        return ApplyToKeys(
            DataKeys.INPUT,
            Compose([
                T.Lambda(lambda x: normalize_tensor(x, self.mean, self.std)),
                # Add other transformations if needed
            ]),
        )


# Binary cross entropy for multi-label classification
def binary_cross_entropy_with_logits(x: Tensor, y: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(x, y.float())

# Define custom Video Classification task
class VC(ClassificationTask):
    backbones: FlashRegistry = FlashRegistry("backbones")

    def __init__(self, num_classes: Optional[int] = None, **kwargs):
        if kwargs.get('labels') and num_classes is None: 
            num_classes = len(kwargs['labels'])
            print(f"Detected num_classes: {num_classes}") 
        super().__init__(**kwargs)
        self.num_classes = num_classes
        print(f"num_classes set in VC: {self.num_classes}") 
        # You can initialize the backbone and head here, as in your original code.


# Проверьте существование файлов
print("Checking files...")
print("Train CSV exists:", os.path.isfile("/home/olga/Pictures/Rehab-app/train_data.csv"))
print("Validation CSV exists:", os.path.isfile("/home/olga/Pictures/Rehab-app/val_data.csv"))
print("Train video directory exists:", os.path.isdir("/home/olga/Pictures/Rehab-app/video_dataset/train"))
print("Validation video directory exists:", os.path.isdir("/home/olga/Pictures/Rehab-app/video_dataset/val"))


print("Model labels:", model.labels)
# Load and prepare data using VideoClassificationData
datamodule = VideoClassificationData.from_csv(
    "video",
    ["HEEL WALKING", "PINCH GRIP", "PLANK", "REVERSE LUNGE", "SCAPULA PROTRACTION", "TREE", "TRICEP DIPS", "TRICEP EXTENSION", "WAITERS BOW", "Y BALANCE"],
    train_file="/home/olga/Pictures/Rehab-app/train_data.csv",
    val_file="/home/olga/Pictures/Rehab-app/val_data.csv",
    train_videos_root="/home/olga/Pictures/Rehab-app/video_dataset/train",
    val_videos_root="/home/olga/Pictures/Rehab-app/video_dataset/val",
    clip_sampler="uniform",
    clip_duration=1,
    batch_size=8,
    num_workers=2,
    transform=VideoClassificationInputTransform(),
)

print("Labels:", datamodule.labels)

# Выводим информацию о содержимом данных
print("Train files:", datamodule.train_dataset)
print("Validation files:", datamodule.val_dataset)


