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
from torch import Tensor
from flash.video import VideoClassificationData


from torchmetrics import Accuracy

import flash
from flash.core.classification import ClassificationTask
from flash.core.data.io.input import DataKeys
from flash.core.registry import FlashRegistry
from flash.core.utilities.compatibility import accelerator_connector
from flash.core.utilities.imports import _PYTORCHVIDEO_AVAILABLE
from flash.core.utilities.providers import _PYTORCHVIDEO
from flash.core.data.io.input_transform import InputTransform
from typing import Callable


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

train_data_path = Path("/home/olga/Pictures/Rehab-app/video_dataset/train")
train_vids = get_files(train_data_path, extensions=[".mp4"])
train_vids[1]



base_path = "/home/olga/Pictures/Rehab-app/video_dataset/train"
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

l = []
for exercise in exercises:
    exercise_path = os.path.join(base_path, exercise)
    for video in os.listdir(exercise_path):
        if video.endswith(('.mp4', '.avi', '.mov')):  # Добавьте другие расширения видео, если необходимо
            video_path = os.path.join(exercise_path, video)
            lab = one_hot_encode(exercise, exercises)
            l.append(lab)
            print(lab)

l = np.array(l)
l = torch.from_numpy(l).float()  # Преобразуем в float для совместимости с PyTorch

print(f"Shape of the labels tensor: {l.shape}")
print(f"Example label: {l[0]}")

train_vids = [str(vid).replace(str(train_data_path) + "/", "") for vid in train_vids]


train_df = pd.DataFrame(
    {
        "video": train_vids,
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
def multiBinary(vids):
    l = []
    for vid in vids:
        exercise = vid.parent.name  # Assuming the parent folder name is the exercise name
        lab = one_hot_encode(exercise, exercises)
        l.append(lab)
    return np.array(l)

# Перемешиваем DataFrame
train_df = train_df.sample(frac=1, random_state=42)

# Выводим первые несколько строк для проверки
print(train_df.head())

# Сохраняем DataFrame в CSV файл
train_df.to_csv("train_data.csv", index=False)
print("DataFrame сохранен в train_data.csv")


def createMultiLabelDf(data_path):
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
            "Y BALANCE": l[:, 9],
        }
    )

    df.to_csv(str(data_path) + "/" + str(data_path.name) + ".csv", index=False)


train_data_path = "/home/olga/Pictures/Rehab-app/video_dataset/train"
val_data_path = "/home/olga/Pictures/Rehab-app/video_dataset/val"
train_csv = createMultiLabelDf(train_data_path)
val_csv = createMultiLabelDf(val_data_path)

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
            Compose(
                [UniformTemporalSubsample(self.temporal_sub_sample), normalize]
                + per_sample_transform
            ),
        )

    def train_per_sample_transform(self) -> Callable:
        per_sample_transform = [RandomCrop(self.image_size, pad_if_needed=True)]

        return ApplyToKeys(
            DataKeys.INPUT,
            Compose(
                [UniformTemporalSubsample(self.temporal_sub_sample), normalize]
                + per_sample_transform
            ),
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
  """Calls BCE with logits and cast the target one_hot (y) encoding to floating point precision."""
  return F.binary_cross_entropy_with_logits(x, y.float())


_VIDEO_CLASSIFIER_BACKBONES = FlashRegistry("backbones")

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
        """Return the module attributes of the model to be frozen."""
        return list(self.backbone.children())

datamodule = VideoClassificationData.from_csv(
    "video",
    ["HEEL WALKING", "PINCH GRIP", "PLANK", "REVERSE LUNGE", "SCAPULA PROTRACTION", "TREE", "TRICEP DIPS", "TRICEP EXTENSION", "WAITERS BOW", "Y BALANCE"],
    train_file="/home/olga/Pictures/Rehab-app/train_data.csv",
    train_videos_root="/home/olga/Pictures/Rehab-app/video_dataset/train",
    val_file="/home/olga/Pictures/Rehab-app/val_data.csv",
    val_videos_root="/home/olga/Pictures/Rehab-app/video_dataset/val",
    transform_kwargs=dict(image_size=(244, 244)),
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
    batch_size=8,
    num_workers=2,
    transform=TransformDataModule(),
    persistent_workers=True,
)

datamodule.labels
["HEEL WALKING",
    "PINCH GRIP",
    "PLANK",
    "REVERSE LUNGE",
    "SCAPULA PROTRACTION",
    "TREE",
    "TRICEP DIPS",
    "TRICEP EXTENSION",
    "WAITERS BOW",
    "Y BALANCE"]

metrics = MultilabelAccuracy(num_labels, threshold=0.5, average=None)
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