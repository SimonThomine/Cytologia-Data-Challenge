import torch
from PIL import Image
import torch.nn.functional as F

from detection.inference.inference_det import YoloInference
from detection.inference.wbf import wbf
 
class EnsembleYolo:
    """Class to perform inference on an ensemble of Yolo models"""
    def __init__(self, models:list[YoloInference],use_probs=False):
        self.models = models
        self.use_probs=use_probs

    def predict(self,img,conf=0.00001,verbose=False): 
        """Ensemble inference to get prediction as a list of boxes, scores and labels"""
        all_boxes=[]
        all_soh=[]

        for model in self.models:
            if model.model_type=="yolov10" or not self.use_probs:
                boxes,scores,preds=model.predict(img,verbose=verbose,conf=conf,mc_nms=False)
            else:
                boxes,scores,preds,probs=model.predict(img,verbose=verbose,conf=conf,mc_nms=True,return_probs=True) 
            all_boxes.append(boxes)

            sohs=[]

            if not self.use_probs or model.model_type=="yolov10":
                for score,pred in zip(scores,preds):
                    soh=F.one_hot(torch.tensor(pred).long(), num_classes=23)*score
                    sohs.append(soh)
            else:
                for prob in probs:
                    prob=torch.tensor(prob)
                    sohs.append(prob)

            all_soh.append(sohs)

        list_all_boxes = [box for boxes in all_boxes for box in boxes]
        list_all_soh = [soh for sohs in all_soh for soh in sohs]

        return wbf(list_all_boxes,list_all_soh)

       
class TtaYolo:
    """Class to perform inference on a single Yolo model with test time augmentation"""
    def __init__(self, model:YoloInference, augmentations=["hflip","vflip"]): 
        self.model = model
        self.augmentations = augmentations

    def predict(self,img,conf=0.00001,verbose=False):
        """Ensemble inference to get prediction as a list of boxes, scores and labels"""
        all_boxes=[]
        all_soh=[]

        if isinstance(img,str):
            img=Image.open(img)
        elif not isinstance(img,Image):
            print("Image should be a path or a PIL image")

        # base transformation
        boxes,scores,preds=self.model.predict(img,verbose=verbose,conf=conf,mc_nms=False) # ! Better with True but wip
        all_boxes.append(boxes)
        
        sohs=[]
        for score,pred in zip(scores,preds):
            soh=F.one_hot(torch.tensor(pred).long(), num_classes=23)*score
            sohs.append(soh)

        all_soh.append(sohs)


        for aug in self.augmentations:
            if aug=="hflip":
                img_flip_h=img.transpose(Image.FLIP_LEFT_RIGHT)
                boxes,scores,preds=self.model.predict(img_flip_h,verbose=verbose,conf=conf,mc_nms=False)# ! Better with True but wip
                for i,box in enumerate(boxes):
                    box[0]=img.width-box[0]
                    box[2]=img.width-box[2]
                    boxes[i]=box[2],box[1],box[0],box[3]

            elif aug=="vflip":
                img_flip_v=img.transpose(Image.FLIP_TOP_BOTTOM)
                boxes,scores,preds=self.model.predict(img_flip_v,verbose=verbose,conf=conf,mc_nms=False)# ! Better with True but wip
                for i,box in enumerate(boxes):
                    box[1]=img.height-box[1]
                    box[3]=img.height-box[3]
                    boxes[i]=box[0],box[3],box[2],box[1]

            all_boxes.append(boxes)
            sohs=[]
            for score,pred in zip(scores,preds):
                soh=F.one_hot(torch.tensor(pred).long(), num_classes=23)*score
                sohs.append(soh)

        all_soh.append(sohs)

        list_all_boxes = [box for boxes in all_boxes for box in boxes]
        list_all_soh = [soh for sohs in all_soh for soh in sohs]

        return wbf(list_all_boxes,list_all_soh)
