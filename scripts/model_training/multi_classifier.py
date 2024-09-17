from types import FunctionType
from typing import Any, Dict, Iterable, List, Optional, Union, Callable
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="flash")

import torch
import os
import pandas as pd
import pytorchvideo
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
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.providers import _PYTORCHVIDEO

import torch
import torchvision.transforms as T
from torchmetrics import Accuracy, F1Score

from flash.core.utilities.types import (
    LOSS_FN_TYPE, LR_SCHEDULER_TYPE, METRICS_TYPE, OPTIMIZER_TYPE
)
from torchvision.transforms import RandomCrop, CenterCrop, Compose

print(hasattr(K, 'VideoSequential'))

def get_files(directory, extensions=['.mp4']):
    files = []
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Warning: {directory} is not a directory")
        return files
    for item in directory.rglob('*'):
        if item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
            print(f"Found file: {item}")
    if not files:
        print(f"No files with extensions {extensions} found in {directory}")
    return files

def one_hot_encode(exercise, exercises):
    return [1 if ex == exercise else 0 for ex in exercises]

def oneHot(vids):
    if not vids:
        print("No videos found.")
        return torch.tensor([])
    
    labels = ["HEEL WALKING", "PINCH GRIP", "PLANK", "REVERSE LUNGE", "SCAPULA PROTRACTION",
              "TREE", "TRICEP DIPS", "TRICEP EXTENSION", "WAITERS BOW", "Y BALANCE"]
    l = []
    for v in vids:
        n = v.parent.name
        lab = [1 if n == label else 0 for label in labels]
        l.append(lab)
    return torch.tensor(l)

def createMultiLabelDf(data_path):
    data_path = Path(data_path)
    vids = get_files(data_path, extensions=[".mp4"])
    print(f"Number of videos found: {len(vids)}")
    
    if not vids:
        print("No videos found. Creating an empty DataFrame.")
        return pd.DataFrame(columns=["video"] + ["HEEL WALKING", "PINCH GRIP", "PLANK", "REVERSE LUNGE", "SCAPULA PROTRACTION",
                                                 "TREE", "TRICEP DIPS", "TRICEP EXTENSION", "WAITERS BOW", "Y BALANCE"])
    
    l = oneHot(vids)
    print(f"Shape of labels tensor: {l.shape}")

    vids = [str(vid.relative_to(data_path)) for vid in vids]

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
    csv_path = data_path / f"{data_path.name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV file created at: {csv_path}")
    return df

def normalize_tensor(tensor: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    if tensor.ndim == 4:  # (batch, channels, height, width)
        return (tensor - mean[None, :, None, None]) / std[None, :, None, None]
    elif tensor.ndim == 5:  # (batch, time, channels, height, width)
        return (tensor - mean[None, None, :, None, None]) / std[None, None, :, None, None]
    else:
        raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

def per_batch_transform_on_device(self) -> Callable:
    return ApplyToKeys(
        DataKeys.INPUT,
        Compose([
            T.Lambda(lambda x: normalize_tensor(x, self.mean, self.std)),
        ]),
    )

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
            ]),
        )

def normalize(x: Tensor) -> Tensor:
    return x / 255.0

class TransformDataModule(InputTransform):
    image_size: int = 256
    temporal_sub_sample: int = 16
    mean: Tensor = torch.tensor([0.45, 0.45, 0.45])
    std: Tensor = torch.tensor([0.225, 0.225, 0.225])
    data_format: str = "BCTHW"
    same_on_frame: bool = False

    def per_sample_transform(self) -> Callable:
        per_sample_transform = [CenterCrop(self.image_size)]
        return Compose(
            [
                ApplyToKeys(
                    DataKeys.INPUT,
                    Compose(
                        [UniformTemporalSubsample(self.temporal_sub_sample), normalize]
                        + per_sample_transform
                    ),
                ),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ]
        )

    def train_per_sample_transform(self) -> Callable:
        per_sample_transform = [RandomCrop(self.image_size, pad_if_needed=True)]
        return Compose(
            [
                ApplyToKeys(
                    DataKeys.INPUT,
                    Compose(
                        [UniformTemporalSubsample(self.temporal_sub_sample), normalize]
                        + per_sample_transform
                    ),
                ),
                ApplyToKeys(DataKeys.TARGET, torch.as_tensor),
            ]
        )

    def per_batch_transform_on_device(self) -> Callable:
        return ApplyToKeys(
            DataKeys.INPUT,
            K.VideoSequential(
                K.Normalize(self.mean, self.std),
                data_format=self.data_format,
                same_on_frame=self.same_on_frame,
            ),
        )

def binary_cross_entropy_with_logits(x: Tensor, y: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(x, y.float())

_VIDEO_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

print(f"PyTorchVideo available: {_PYTORCHVIDEO_AVAILABLE}")

if _PYTORCHVIDEO_AVAILABLE:
    from pytorchvideo.models import hub
    for fn_name in dir(hub):
        if "__" not in fn_name:
            fn = getattr(hub, fn_name)
            if isinstance(fn, FunctionType):
                _VIDEO_CLASSIFIER_BACKBONES(fn=fn, providers=_PYTORCHVIDEO)

class VC(ClassificationTask):
    backbones: FlashRegistry = _VIDEO_CLASSIFIER_BACKBONES
    required_extras = "video"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        multi_label: bool = False,
        labels: Optional[List[str]] = None,
        backbone: Union[str, nn.Module] = "x3d_xs",
        backbone_kwargs: Optional[Dict] = None,
        pretrained: bool = True,
        loss_fn: LOSS_FN_TYPE = binary_cross_entropy_with_logits,
        optimizer: OPTIMIZER_TYPE = "Adam",
        lr_scheduler: LR_SCHEDULER_TYPE = None,
        metrics: METRICS_TYPE = Accuracy(),
        learning_rate: Optional[float] = None,
        head: Optional[Union[FunctionType, nn.Module]] = None,
    ):
        self.save_hyperparameters()

        if labels is not None and num_classes is None:
            num_classes = len(labels)

        super().__init__(
            model=None,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            learning_rate=learning_rate,
            num_classes=num_classes,
            labels=labels,
            multi_label=multi_label,
        )

        if not backbone_kwargs:
            backbone_kwargs = {}

        backbone_kwargs["pretrained"] = (
            True if (flash._IS_TESTING and torch.cuda.is_available()) else pretrained
        )
        backbone_kwargs["head_activation"] = None

        if isinstance(backbone, nn.Module):
            self.backbone = backbone
        elif isinstance(backbone, str):
            self.backbone = self.backbones.get(backbone)(**backbone_kwargs)
            num_features = self.backbone.blocks[-1].proj.out_features
        else:
            raise ValueError(
                f"backbone should be either a string or a nn.Module. Found: {backbone}"
            )

        self.head = head or nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_features, num_classes),
        )

    def on_train_start(self) -> None:
        if accelerator_connector(self.trainer).is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.data
            encoded_dataset._video_sampler = DistributedSampler(
                encoded_dataset._labeled_videos
            )
        super().on_train_start()

    def on_train_epoch_start(self) -> None:
        if accelerator_connector(self.trainer).is_distributed:
            encoded_dataset = self.trainer.train_dataloader.loaders.dataset.data
            encoded_dataset._video_sampler.set_epoch(self.trainer.current_epoch)
        super().on_train_epoch_start()

    def step(self, batch: Any, batch_idx: int, metrics) -> Any:
        return super().step(
            (batch[DataKeys.INPUT], batch[DataKeys.TARGET]), batch_idx, metrics
        )

    def forward(self, x: Any) -> Any:
        x = self.backbone(x)
        if self.head is not None:
            x = self.head(x)
        return x

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        predictions = self(batch[DataKeys.INPUT])
        batch[DataKeys.PREDS] = predictions
        return batch

    def modules_to_freeze(
        self,
    ) -> Union[nn.Module, Iterable[Union[nn.Module, Iterable]]]:
        return list(self.backbone.children())

# Main execution
print("Checking files...")
print("Train CSV exists:", os.path.isfile("/home/olga/Pictures/Rehab-app/train_data.csv"))
print("Validation CSV exists:", os.path.isfile("/home/olga/Pictures/Rehab-app/val_data.csv"))
print("Train video directory exists:", os.path.isdir("/home/olga/Pictures/Rehab-app/video_dataset/train"))
print("Validation video directory exists:", os.path.isdir("/home/olga/Pictures/Rehab-app/video_dataset/val"))

train_data_path = Path("/home/olga/Pictures/Rehab-app/train")
val_data_path = Path("/home/olga/Pictures/Rehab-app/val")

print("Creating train CSV...")
train_csv = createMultiLabelDf(train_data_path)
print("Train CSV shape:", train_csv.shape)

print("\nCreating validation CSV...")
val_csv = createMultiLabelDf(val_data_path)
print("Validation CSV shape:", val_csv.shape)

print("\nChecking CSV contents:")
print("Train CSV:")
print(train_csv.head())
print("\nValidation CSV:")
print(val_csv.head())

datamodule = VideoClassificationData.from_csv(
    "video",
    ["HEEL WALKING", "PINCH GRIP", "PLANK", "REVERSE LUNGE", "SCAPULA PROTRACTION", "TREE", "TRICEP DIPS", "TRICEP EXTENSION", "WAITERS BOW", "Y BALANCE"],
    train_file=str(train_data_path / "train.csv"),
    val_file=str(val_data_path / "val.csv"),
    train_videos_root=str(train_data_path),
    val_videos_root=str(val_data_path),
    clip_sampler="uniform",
    clip_duration=1,
    batch_size=8,
    num_workers=2,
    transform=VideoClassificationInputTransform(),
)

print("Labels:", datamodule.labels)
print("Train files:", datamodule.train_dataset)
print("Validation files:", datamodule.val_dataset)

metrics = F1Score(num_labels=datamodule.num_classes, task="multilabel", top_k=1)


head = nn.Sequential(
    nn.Flatten(start_dim=1, end_dim=-1),
    nn.Linear(in_features=400, out_features=7, bias=True),
    nn.Sigmoid(),
)

model = VC(
    backbone="x3d_m",
    labels=datamodule.labels,
    metrics=metrics,
    loss_fn=binary_cross_entropy_with_logits,
    head=head,
    multi_label=datamodule.multi_label,
    pretrained=True,
)


trainer = flash.Trainer(
    max_epochs=2,
    accelerator="gpu",
    devices=1,
)
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

