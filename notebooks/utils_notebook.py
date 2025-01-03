from PIL import Image,ImageDraw 
import numpy as np
import pandas as pd
import os
import shutil
from tqdm import tqdm

def calculate_iou_max(yolo_boxes, box):
    def compute_iou(box1, box2):
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    iou_values = [(compute_iou(box, yolo_box), yolo_box, idx) for idx, yolo_box in enumerate(yolo_boxes)]
    max_iou, best_box, best_idx = max(iou_values, key=lambda x: x[0], default=(0, None, None))
    
    return max_iou, best_box,best_idx

def filter_boxes(occurences,boxes,scores,labels):
    while len(occurences) < len(boxes):
        min_score = min(scores)
        idx =scores.index(min_score)
        boxes.pop(idx)
        scores.pop(idx)
        labels.pop(idx)
    return boxes,scores,labels


# New image curation with suppression of boxes with low iou ? 
def get_best_yolo_box(gt_box,yolo_boxes,yolo_labels,yolo_scores,mode="msk_iou"):

    iou,yolo_box,idx=calculate_iou_max(yolo_boxes,gt_box)
    if mode=="msk_iou":
        if iou<0.3:
            return None,yolo_boxes,yolo_labels,yolo_scores
    yolo_boxes=np.delete(yolo_boxes,idx,axis=0)
    yolo_labels=np.delete(yolo_labels,idx,axis=0)
    yolo_scores=np.delete(yolo_scores,idx,axis=0)

    return yolo_box,yolo_boxes,yolo_labels,yolo_scores

# msk_iou : keep only the gt boxes with high iou (>0.3) and add the others as msk

def curate_image(boxes,yolo_output,img_path,classes,df,new_data,mode="msk_iou"):
    name=img_path.split("/")[-1]
    yolo_boxes,yolo_scores,yolo_labels=yolo_output

    for gt_box,gt_label in zip(boxes,classes):
        if len(yolo_boxes)>0:
            yolo_box,yolo_boxes,yolo_labels,yolo_scores=get_best_yolo_box(gt_box,yolo_boxes,yolo_labels,yolo_scores,mode=mode)

            if yolo_box is not None:     
                yolo_box=[int(x) for x in yolo_box]

                gt_x1, gt_y1, gt_x2, gt_y2 = gt_box  

                df.loc[(df['NAME'] == name) & 
                (df['x1'] == gt_x1) & 
                (df['y1'] == gt_y1) & 
                (df['x2'] == gt_x2) & 
                (df['y2'] == gt_y2), 
                ['x1', 'y1', 'x2', 'y2', 'class']] = [yolo_box[0], yolo_box[1], yolo_box[2], yolo_box[3], gt_label]
            else:
                cls="msk"
                new_data.append({
                    'NAME': name,
                    'x1': gt_box[0],
                    'y1': gt_box[1],
                    'x2': gt_box[2],
                    'y2': gt_box[3],
                    'class': cls,
                    })

    for yolo_box, _,_ in zip(yolo_boxes, yolo_labels,yolo_scores):
        cls="msk"
        yolo_box=[int(x) for x in yolo_box]
        new_data.append({
            'NAME': name,
            'x1': yolo_box[0],
            'y1': yolo_box[1],
            'x2': yolo_box[2],
            'y2': yolo_box[3],
            'class': cls,
            })
    

def add_soft_label(yolo_output,img_path,itos,new_data,threshold=0.5):
    name=img_path.split("/")[-1]
    yolo_boxes,yolo_scores,yolo_labels=yolo_output

    for box,label,score in zip(yolo_boxes,yolo_labels,yolo_scores):
        box=[int(x) for x in box]
        if score<threshold:
            cls="msk"
        else:
            cls=itos[label]
        
        new_data.append({
                'NAME': name,
                'x1': box[0],
                'y1': box[1],
                'x2': box[2],
                'y2': box[3],
                'class': cls
                })

# Different types of masks : black, mean, interpolated, cutpaste
def add_msk(img_path, image_annotations):
    """Add black masks on an image at the bounding boxes coordinates with class 'msk', masking only non-overlapping parts"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
 

    non_msk_boxes = image_annotations[image_annotations['class'] != "msk"][['x1', 'y1', 'x2', 'y2']].values

    for _, row in image_annotations.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        if row['class'] == "msk":
            current_box = [x1, y1, x2, y2]
            
            parts_to_mask = [current_box]

            for other_box in non_msk_boxes:
                overlap = get_overlap(current_box, other_box)
                if overlap:
                    new_parts = []
                    for part in parts_to_mask:
                        new_parts.extend(split_box(part, overlap))
                    parts_to_mask = new_parts
            for part in parts_to_mask:
                draw.rectangle(part, fill="black")                    
    return img

def split_box(box, overlap):
    """Find the non-overlapping parts of a box with another box"""
    x1, y1, x2, y2 = box
    ox1, oy1, ox2, oy2 = overlap

    new_boxes = []

    if y1 < oy1:  # Haut
        if x1<=x2 and y1<=oy1:
            new_boxes.append([x1, y1, x2, oy1])
    if y2 > oy2:  # Bas
        if x1<=x2 and oy2<=y2:
            new_boxes.append([x1, oy2, x2, y2])
    if x1 < ox1:  # Gauche
        if x1<=ox1 and max(y1, oy1)<=min(y2, oy2):
            new_boxes.append([x1, max(y1, oy1), ox1, min(y2, oy2)])
    if x2 > ox2:  # Droite
        if ox2<=x2 and max(y1, oy1)<=min(y2, oy2):
            new_boxes.append([ox2, max(y1, oy1), x2, min(y2, oy2)])

    return new_boxes


def get_overlap(box1, box2):
        """Retourne la zone de chevauchement entre deux rectangles."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_overlap = max(0, min(x2_1, x2_2) - max(x1_1, x1_2))
        y_overlap = max(0, min(y2_1, y2_2) - max(y1_1, y1_2))
        
        if x_overlap > 0 and y_overlap > 0:
            return [
                max(x1_1, x1_2), max(y1_1, y1_2), 
                min(x2_1, x2_2), min(y2_1, y2_2)
            ]
        return None


def split_dataset_from_csv(new_data_path,data_path, csv_file):
    # Charger les annotations depuis le CSV
    annotations_df = pd.read_csv(csv_file)
    all_images = annotations_df['NAME'].unique()

    img_dir = os.path.join(new_data_path, 'images')
    label_dir = os.path.join(new_data_path, 'labels')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    tqdm_imgs = tqdm(all_images)
    for image_name in tqdm_imgs:
        image_annotations = annotations_df[annotations_df['NAME'] == image_name]
        needs_modif=False
        for _, row in image_annotations.iterrows():
            if row['class']=="msk":
                needs_modif=True
        
        if needs_modif:
            # handle gestion of image with msk
            img_path = os.path.join(data_path, image_name)
            modified_img=add_msk(img_path,image_annotations)
            # This is PIL
            modified_img.save(os.path.join(img_dir, image_name))
        else:
            img_path = os.path.join(data_path, image_name)
            shutil.copy(img_path, os.path.join(img_dir, image_name))
        label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        create_yolo_annotations(image_annotations, img_dir, image_name, label_path) # 
    


def create_yolo_annotations(image_annotations, target_dir, image_name, label_path):
    stoi={'B':0, 'BA':1, 'EO':2, 'Er':3, 'LAM3':4, 'LF':5, 'LGL':6, 'LH_lyAct':7, 'LLC':8, 'LM':9, 'LY':10, 'LZMG':11, 'LyB':12, 'Lysee':13, 'M':14, 'MBL':15, 'MM':16, 'MO':17, 'MoB':18, 'PM':19, 'PNN':20, 'SS':21, 'Thromb':22}
    """Crée les annotations au format YOLO pour une image donnée."""
    img_path = os.path.join(target_dir, image_name)
    with Image.open(img_path) as img:
        img_width, img_height = img.size
    
    #yolo_annotation_file = os.path.join(target_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    
    with open(label_path, 'w') as f:
        for _, row in image_annotations.iterrows():
            if row['class']=="msk":
                continue
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            class_id = stoi[row['class']]
            
            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")