# Implementation of weighted boxes fusion (WBF) algorithm
import numpy as np
from collections import defaultdict

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

def fuse_box(cluster):
    boxes, sohs= cluster["boxes"], cluster["soh"]

    label_score_sum = defaultdict(float) 
    for soh in sohs:
        for label, score in enumerate(soh[:]):
             label_score_sum[label] += score  
        
    best_label = max(label_score_sum, key=label_score_sum.get)

    filtered = [(box, soh) for box, soh in zip(boxes, sohs) if np.argmax(soh) == best_label]

    not_filtered = [(box, soh) for box, soh in zip(boxes, sohs)]    
    x1_mean = sum(box[0] for box, _ in not_filtered) / len(not_filtered)
    y1_mean = sum(box[1] for box, _ in not_filtered) / len(not_filtered)
    x2_mean = sum(box[2] for box, _ in not_filtered) / len(not_filtered)
    y2_mean = sum(box[3] for box, _ in not_filtered) / len(not_filtered)

    soh_mean = sum(soh for _, soh in filtered) / len(boxes)

    return {"box": [round(x1_mean), round(y1_mean), round(x2_mean), round(y2_mean)], "soh": soh_mean}
    
def wbf(boxes,sohs):
    # Sort given boxes by their scores
    max_scores = np.max(sohs,axis=1)
    sorted_indices = np.argsort(max_scores)[::-1]

    boxes=np.array(boxes)
    sohs=np.array(sohs)

    boxes = boxes[sorted_indices] # B in paper
    sohs = sohs[sorted_indices] 

    # Elements of clusters are Dict of boxes and sohs
    clusters=[] #L in paper
    # Elements of fused_boxes are Dict of box and soh
    fused_boxes=[] #F in paper

    for box, soh in zip(boxes, sohs):
        associated=False
        for i,fused_box in enumerate(fused_boxes):
            f_box=fused_box["box"] 
            iou=compute_iou(box, f_box)
            if iou>0.4: # was 0.55 in the paper
                clusters[i]["boxes"].append(box)
                clusters[i]["soh"].append(soh)
                associated=True
                # Computer new fused box
                fused_boxes[i]=fuse_box(clusters[i])
                break

        if not associated:
            clusters.append({"boxes":[box],"soh":[soh]})
            fused_boxes.append({"box":box,"soh":soh})

    # Return the boxes, scores and labels, np array or not ?
    boxes = np.array([fused_box["box"] for fused_box in fused_boxes])
    scores = np.array([np.max(fused_box["soh"]) for fused_box in fused_boxes])
    labels = np.array([np.argmax(fused_box["soh"]) for fused_box in fused_boxes])

    return boxes, scores, labels