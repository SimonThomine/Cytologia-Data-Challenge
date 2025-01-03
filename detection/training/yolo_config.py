from dataclasses import dataclass,field
from typing import Optional
from utils import set_seed

@dataclass
class YoloConfig:
    """Training configuration dataclass
    
    Attributes:
        dataset (str): Name of the dataset to use
        epochs (int): Number of epochs to train the model
        batch_size (int): Batch size for training
        workers (int): Number of workers to use for the dataloader
        img_size (int): Image size to use
        nc (int): Number of classes (either 1 for one class detector or 23 for multi class)
        classes (list[str]): List of classes (names corresponding to the class index) => kind of redundant with nc, only classes might be enough
        backbone (str): Backbone to use
        identifier (str): Identifier for the model
        device (str): Device to use for training
        seed (Optional[int]): Seed to use for reproducibility, if None no seed is set
        val_split (float): Validation split to use, for cross validation, this parameter is not used
        folds (int): Number of folds for cross validation, if 1, no cross validation is used
        data_augmentation (dict): Dictionary of data augmentation parameters (the one you want to change from the default)
    """

    dataset: str
    epochs: int = 10
    batch_size: int =64
    workers: int = 4
    img_size: int = 384
    nc: int = 1
    classes: list = field(default_factory=list)
    backbone: str = "yolo11n.pt"
    identifier: str = "test"
    device: str = "cuda"
    seed: Optional[int] = 42
    val_split: float = 0.2
    folds: int = 1
    data_augmentation: dict = field(default_factory=dict)

    def __post_init__(self):
        set_seed(self.seed)