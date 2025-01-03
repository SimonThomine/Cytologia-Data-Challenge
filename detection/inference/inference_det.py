import os
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.ops import nms
from ultralytics.data.augment import LetterBox
import torchvision.transforms as T
from PIL import Image
from detection.inference.custom_nms import non_max_suppression_modified

class YoloInference():
    def __init__(self,weights,device="cuda",verbose=False,load_model=True):
        self.device=device
        self.load_model=load_model
        self.weights=weights
        self.model=None
        if load_model:
            self.yolo=YOLO(weights,verbose=verbose).to(self.device)
            self.model=self.yolo.model

        self.model_type=None
        if self.weights.find("yolov10")!=-1:
            self.model_type="yolov10"
        elif self.weights.find("yolo11")!=-1:
            self.model_type="yolo11"
        elif self.weights.find("yolov8")!=-1:
            self.model_type="yolov8"


        self.letter_box=LetterBox((384,384))
        self.transform=T.Compose([T.ToTensor()])  

        self.conf_mat=None

        # load the conf matrix for eventual ensembling if it exists
        conf_path=os.path.dirname(os.path.dirname(os.path.dirname(weights)))+"/conf_matrix.npy"
        if os.path.exists(conf_path):
           self.conf_mat=np.load(conf_path)
    
    def predict(self,img,verbose=False,conf=0.05,mc_nms=True,return_probs=False):
        """Inference on image to get a list of boxes, scores and labels"""
        if self.model_type=="yolov10" or not return_probs:
            return self.predict_base(img,verbose=verbose,conf=conf,mc_nms=mc_nms)
        elif self.model_type=="yolo11" or self.model_type=="yolov8": 
            return self.predict_probs(img,verbose=verbose,conf=conf,mc_nms=mc_nms)
        else:
            print("Model type not recognized")
            return None

    def predict_probs(self,img,verbose=False,conf=0.05,mc_nms=True):
        """Inference to get probs of detected classes as a supplementary output, this is used for ensembling"""
        if not self.load_model:
            print("LOADING MODEL")
            self.yolo=YOLO(self.weights,verbose=verbose).to(self.device)
            self.model=self.yolo.model

        # Image loading
        if isinstance(img,str):
            img=Image.open(img)
        img_np=self.letter_box(image=np.array(img))
        transform=T.Compose([T.ToTensor()]) # 
        img_tensor=transform(img_np).unsqueeze(0).to("cuda")

        # inference
        prediction,_ = self.model(img_tensor)  
        
        outputs,outputs_oh=non_max_suppression_modified(prediction, conf_thres=conf, iou_thres=0.4,agnostic=mc_nms)
        outputs=outputs[0]
        outputs_oh=outputs_oh[0]
        

        width,height=img.size
        boxes,scores,labels,probs=[],[],[],[]
        for out,out_oh in zip(outputs,outputs_oh):
            out=out.cpu().numpy()
            out_oh=out_oh.cpu().numpy()
            box=out[:4]
            x1,y1,x2,y2=box
            # Rescale to fit the original image size
            x1,y1,x2,y2=max(0,x1*width/384),max(0,y1*height/384),min(width,x2*width/384),min(height,y2*height/384)
            score=out[4]
            label=out[5]
            prob=out_oh[4:]
            boxes.append((np.round(x1).astype(int),np.round(y1).astype(int),np.round(x2).astype(int),np.round(y2).astype(int)))
            scores.append(score)
            labels.append(label)
            probs.append(prob)
        
        boxes = np.asarray(boxes)
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        probs = np.asarray(probs)

        
        if not self.load_model:
            self.unload_model()

        return boxes,scores,labels,probs
    

    def predict_base(self,img,verbose=False,conf=0.05,mc_nms=True):
        """Classic inference to get boxes, scores and labels"""
        if not self.load_model:
            print("LOADING MODEL")
            self.yolo=YOLO(self.weights,verbose=verbose).to(self.device)

        result=self.yolo.predict(img,verbose=verbose,iou=0.4,conf=conf) # iou threshold for non-max suppression

        # may deactivate mc_nms when using wbf in ensemble/tta
        if mc_nms:
            kept=nms(result[0].boxes.xyxy,result[0].boxes.conf,0.4) 
            boxes=result[0].boxes.xyxy[kept]
            scores=result[0].boxes.conf[kept]
            labels=result[0].boxes.cls[kept]
        else:
            boxes = result[0].boxes.xyxy
            scores = result[0].boxes.conf
            labels = result[0].boxes.cls
        

        boxes = boxes.cpu().numpy()
        for i,box in enumerate(boxes):
            x1,y1,x2,y2=box 
            boxes[i]=np.round(x1).astype(int),np.round(y1).astype(int),np.round(x2).astype(int),np.round(y2).astype(int)


        scores=scores.cpu().numpy()
        labels=labels.cpu().numpy()
            
        if not self.load_model:
            self.unload_model()

        return boxes,scores,labels

    def unload_model(self):
        """Unload the model to free up memory."""
        self.yolo = None
        self.model = None
        torch.cuda.empty_cache() 

