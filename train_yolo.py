from detection import YoloConfig, YoloTrainer
from utils import get_device

device=get_device()

classes=['B', 'BA', 'EO', 'Er', 'LAM3', 'LF', 'LGL', 'LH_lyAct', 'LLC', 'LM', 'LY', 'LZMG', 'LyB', 'Lysee', 'M', 'MBL', 'MM', 'MO', 'MoB', 'PM', 'PNN', 'SS', 'Thromb']

data_augmentation = {"mixup":0.0}

conf=YoloConfig(
    dataset="/absolute/path/to/your/dataset", 
    backbone="yolo11n.pt",
    img_size=384, # multiple of 32
    identifier="test",
    nc=23,
    classes=classes,
    device=device,
    epochs=100, 
    batch_size=64,
    seed=42,
    folds=1, 
    val_split=0.05,
    data_augmentation=data_augmentation,
)
trainer=YoloTrainer(conf)
trainer.train()

