from ast import literal_eval
from collections import defaultdict
import json
import os

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
from copy import deepcopy
from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import profile


def split_dataset(csv_path: os.PathLike, split_rate: float = 0.2) -> None:
    
    del_data_list = ['3fe6394cd', '41457a646']
    grouped_list = []

    root_dir = os.path.dirname(csv_path)
    df = pd.read_csv(csv_path)
    df = df.sample(frac=1).reset_index(drop=True)
    grouped = df.groupby(by='image_id')
    for i, group in grouped:
        if i in del_data_list: continue
        grouped_list.append(group)
    
    split_point = int(split_rate * len(grouped))
    train_ids = grouped_list[split_point:]
    test_ids = grouped_list[:split_point]

    train_df = pd.concat(train_ids)
    test_df = pd.concat(test_ids)
    train_df.to_csv(os.path.join(root_dir, 'train_answer.csv'), index=False)
    test_df.to_csv(os.path.join(root_dir, 'test_answer.csv'), index=False)


class MeanAveragePrecision:
    def __init__(self, csv_path: os.PathLike) -> None:
        self.id_csv2coco = {}
        json_path = self.to_coco(csv_path)
        self.coco_gt = COCO(json_path)

        with open(json_path, "r") as f:
            json_data = json.load(f)

        self.images_data = json_data['images']

        self.detections = []

    def to_coco(self, csv_path: os.PathLike) -> os.PathLike:
        df = pd.read_csv(csv_path)
        
        grouped = df.groupby(by='image_id')
        grouped_dict = {image_id: group for image_id, group in grouped}
        
        res = defaultdict(list)

        n_id = 0
        for image_id, (file_name, group) in enumerate(grouped_dict.items()):
            res['images'].append({
                'id': image_id,
                'width': 1024,
                'height': 1024,
                'file_name': f'{file_name}.jpg',
            })

            self.id_csv2coco[file_name] = image_id

            for _, row in group.iterrows():
                x1, y1, w, h = literal_eval(row['bbox'])
                res['annotations'].append({
                    'id': n_id,
                    'image_id': image_id,
                    'category_id': 1,
                    'area': w * h,
                    'bbox': [x1, y1, w, h],
                    'iscrowd': 0,
                })
                n_id += 1

        res['categories'].extend([{'id': 1, 'name': 'wheat'}])
            
        root_dir = os.path.split(csv_path)[0]
        save_path = os.path.join(root_dir, 'coco_annotations.json')
        with open(save_path, 'w') as f:
            json.dump(res, f)

        return save_path
    
    def update(self, preds, image_ids, image_size):
        for p, image_id in zip(preds, image_ids):

            for image_data in self.images_data:
                if image_data['file_name'].replace(".jpg", "") == image_id:
                    origin_w, origin_h = image_data['width'], image_data['height']
                    w_rate = origin_w / image_size
                    h_rate = origin_h / image_size
                    break

            p['boxes'][:, 0] *= w_rate
            p['boxes'][:, 1] *= h_rate
            p['boxes'][:, 2] *= w_rate
            p['boxes'][:, 3] *= h_rate
            p['boxes'][:, 2] = p['boxes'][:, 2] - p['boxes'][:, 0]
            p['boxes'][:, 3] = p['boxes'][:, 3] - p['boxes'][:, 1]
            p['boxes'] = p['boxes'].cpu().numpy()

            p['scores'] = p['scores'].cpu().numpy()
            p['labels'] = p['labels'].cpu().numpy()

            image_id = self.id_csv2coco[image_id]
            for box, cat, score in zip(*p.values()):
                self.detections.append({
                    'image_id': image_id,
                    'category_id': cat,
                    'bbox': box.tolist(),
                    'score': score
                })

    def reset(self):
        self.detections = []

    def compute(self):
        coco_dt = self.coco_gt.loadRes(self.detections)

        coco_eval = COCOeval(self.coco_gt, coco_dt, 'bbox')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        return coco_eval.stats
    
    
def check_train_batch_size(model, imgsz=640):
    

    with torch.cuda.amp.autocast():
        return autobatch(deepcopy(model).eval(), imgsz)  # compute optimal batch size
    

def autobatch(model, imgsz=640, fraction=0.80, batch_size=DEFAULT_CFG.batch):
    
    
    
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz}")
    device = next(model.parameters()).device

   
    if device.type == "cpu":
        LOGGER.info(f"{prefix}CUDA not detected, using default CPU batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    gb = 1 << 30  
    d = str(device).upper()
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz, dtype=torch.float16) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)

       
        y = [x[2] for x in results if x]
        p = np.polyfit(batch_sizes[:len(y)], y, deg=1)
        b = int((f * fraction - p[1]) / p[0])
        if None in results:  
            i = results.index(None)
            if b >= batch_sizes[i]:
                b = batch_sizes[max(i - 1, 0)]
        if b < 1 or b > 1024: 
            b = batch_size
            LOGGER.info(f"{prefix}WARNING ⚠️ CUDA anomaly detected, using default batch-size {batch_size}.")

        fraction = (np.polyval(p, b) + r + a) / t 
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.info(f"{prefix}WARNING ⚠️ error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
        