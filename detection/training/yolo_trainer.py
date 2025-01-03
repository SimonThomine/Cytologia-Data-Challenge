import os
import torch
import numpy as np
from ultralytics import YOLO
from sklearn.model_selection import KFold

class YoloTrainer():
    """ Trainer for yolo"""
    def __init__(self, conf): 
        
        self.conf=conf
        if "fold" in conf.dataset:
            dataset_name=conf.dataset.split("/")[-2]+"/"+conf.dataset.split("/")[-1]
        else:
            dataset_name=conf.dataset.split("/")[-1]

        if os.path.basename(conf.backbone)==conf.backbone:
            self.backbone_filt=conf.backbone.split(".")[-2]
            self.model_name=conf.backbone.split(".")[-2] if "yolo11" in conf.backbone else f'{conf.backbone.split(".")[-2]}.{conf.backbone.split(".")[-1]}'
        else:
            split=conf.backbone.split("/")
            for s in split:
                if "yolo11" in s:
                    self.backbone_filt=s
                    self.model_name=s
                    break
                    
        if conf.folds==1:
            self.model_dir = f"models/detection/{dataset_name}/{self.backbone_filt}/{self.conf.img_size}/{self.conf.identifier}"
        else:
            self.model_dir = f"models/detection/{dataset_name}/{self.backbone_filt}/{self.conf.img_size}/{self.conf.identifier}_cv"

        self.resume=os.path.exists(f'{self.model_dir}/train/weights/best.pt')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)

        if self.conf.folds==1:
            self.data_yaml=self.create_data_yaml() # single file
        else:
            self.data_yaml=self.create_data_yaml_with_cv() # multiple files (one per fold)
        
    

    def create_data_yaml(self):
        """Create the yaml conf file for the Yolo model"""
        data_yaml_path = f"{self.model_dir}/data.yaml"

        images=os.listdir(self.conf.dataset+"/images")
        images=[self.conf.dataset+"/images/"+image for image in images if image.endswith(".jpg")]
        # shuffle
        np.random.shuffle(images)
        # split train/val
        train_len=int(len(images)*(1-self.conf.val_split))
        train_images=images[:train_len]
        val_images=images[train_len:]

        train_file = "train.txt"
        val_file =  "val.txt"
        with open(f'{self.conf.dataset}/{train_file}', "w") as f:
            f.writelines(f"{img}\n" for img in train_images)
        with open(f'{self.conf.dataset}/{val_file}', "w") as f:
            f.writelines(f"{img}\n" for img in val_images)

        # also save txt in the model folder (for later use since the other .txt files will be deleted when training another yolo model)
        with open(f'{self.model_dir}/{train_file}', "w") as f:
            f.writelines(f"{img}\n" for img in train_images)
        with open(f'{self.model_dir}/{val_file}', "w") as f:
            f.writelines(f"{img}\n" for img in val_images)

        data_yaml_content = f"""
        train: {f'{self.conf.dataset}/{train_file}'}
        val: {f'{self.conf.dataset}/{val_file}'}

        nc: {self.conf.nc}
        names: {self.conf.classes}
        """
        
        with open(data_yaml_path, 'w') as file:
            file.write(data_yaml_content)
        
        return data_yaml_path
        
    def create_data_yaml_with_cv(self, identifier=""):
        """Create multiple yaml conf files for cross-validation"""
        data_yaml_paths = []  # Pour stocker les chemins des fichiers YAML
        images = os.listdir(self.conf.dataset + "/images")
        images = [self.conf.dataset + "/images/" + image for image in images if image.endswith(".jpg")]

        # Shuffle les données avant de les utiliser
        np.random.shuffle(images)
        
        # Instanciation de KFold
        kf = KFold(n_splits=self.conf.folds, shuffle=True, random_state=self.conf.seed)
        
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(images)):
            train_images = [images[i] for i in train_idx]
            val_images = [images[i] for i in val_idx]

            train_file = f"train_fold{fold_idx}.txt"
            val_file = f"val_fold{fold_idx}.txt"
            
            with open(f'{self.conf.dataset}/{train_file}', "w") as f:
                f.writelines(f"{img}\n" for img in train_images)
            with open(f'{self.conf.dataset}/{val_file}', "w") as f:
                f.writelines(f"{img}\n" for img in val_images)
            
            # also save txt in the model folder (for later use since the other .txt files will be deleted when training another yolo model)
            with open(f'{self.model_dir}/{train_file}', "w") as f:
                f.writelines(f"{img}\n" for img in train_images)
            with open(f'{self.model_dir}/{val_file}', "w") as f:
                f.writelines(f"{img}\n" for img in val_images)

            data_yaml_content = f"""
            train: {f'{self.conf.dataset}/{train_file}'}
            val: {f'{self.conf.dataset}/{val_file}'}

            nc: {self.conf.nc}
            names: {self.conf.classes}
            """
            data_yaml_path = f"{self.model_dir}/data_fold{fold_idx}{identifier}.yaml"
            with open(data_yaml_path, 'w') as file:
                file.write(data_yaml_content)
            
            data_yaml_paths.append(data_yaml_path)
        
        return data_yaml_paths
    
    def load_model(self,identifier=None):
        """Load the Yolo Ultralytics model"""
        if self.resume:
            if identifier:
                self.model=YOLO(f"{self.model_dir}/{identifier}/train/weights/best.pt",verbose=False)
            else:
                self.model=YOLO(f"{self.model_dir}/train/weights/best.pt",verbose=False)
        else:
            self.model=YOLO(self.model_dir+"/"+self.model_name,verbose=False)
        
    def check_resume(self,model_dir_compl):
        """Check if the model has already been trained for some epochs and needs to be resumed"""
        if not os.path.exists(model_dir_compl) or self.dir_created:
            os.makedirs(model_dir_compl, exist_ok=True)
            self.resume=False
        else :
            self.resume=True

    def train(self):
        """Train the model"""
        if self.conf.folds==1:
            self.load_model()
            self.model.train(
                data=self.data_yaml,
                epochs=self.conf.epochs,
                imgsz=self.conf.img_size,
                batch=self.conf.batch_size,
                project=self.model_dir,
                device=self.conf.device,
                exist_ok=True,
                verbose=False,
                plots=True,
                resume=self.resume,
                **self.conf.data_augmentation
                )
        else:
            for i, data_yaml in enumerate(self.data_yaml):
                self.check_resume(self.model_dir+f"/fold_{i}")
                self.load_model(identifier=f"fold_{i}")
                try:
                    self.model.train(
                        data=data_yaml,
                        epochs=self.conf.epochs,
                        imgsz=self.conf.img_size,
                        batch=self.conf.batch_size,
                        project=f"{self.model_dir}/fold_{i}",
                        device=self.conf.device,
                        exist_ok=True,
                        verbose=False,
                        plots=True,
                        resume=self.resume,
                        **self.conf.data_augmentation
                        )
                    self.model=None
                    torch.cuda.empty_cache()
                except AssertionError as e:
                    if str(e).startswith("models/detection"):
                        print("Training is already finished for this fold. Nothing to resume, proceeding to the next fold.")
                    else:
                        raise e
                
                
    def test(self):
        self.load_weights()
        self.model.predict(source=f"{self.conf.dataset}/val", save=True,iou=0.5)

    def val(self):
        self.load_weights()
        self.model.val(source=f"{self.conf.dataset}/val")

    def load_weights(self):
        """Load the previously saved model weights"""
        if os.path.exists(self.model_dir+"/train/weights/best.pt"):
            self.model=YOLO(self.model_dir+"/train/weights/best.pt")
        else:
            raise Exception(f"Model {self.model_dir}/train/weights/best.pth not found.")
