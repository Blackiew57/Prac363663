import argparse
import os
import random
import shutil
import wandb

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import Dataset

from src.dataset import WheatDataset, get_transform
from src.model import DetectionModel



parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", help="device for learning")
parser.add_argument("--image_size", default=1024, help="int for image resize")
args = parser.parse_args()


def visualize_predictions(testset: Dataset, device: str, model: nn.Module, save_dir: os.PathLike, conf_thr: float = 0.5, n_images: int = 10) -> None:

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        shutil.rmtree(save_dir)
        os.makedirs(save_dir)

    classes = ['wheat']

    model.eval()
    indices = random.choices(range(len(testset)), k=n_images)
    for i in tqdm(indices):
        image, _, image_id = testset[i]
        image = [image.to(device, dtype=torch.float32)]

        pred = model(image)

        image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
        pred = {k: v.detach().cpu() for k, v in pred[0].items()}

        plt.imshow(image)
        ax = plt.gca()

        for box, category_id, score in zip(*pred.values()):
            if score >= conf_thr:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                category_id = category_id.item()

                rect = patches.Rectangle(
                    (x1, y1),
                    w, h,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='none'
                )
                ax.add_patch(rect)

                text = ax.text(
                    x1, y1,
                    f'{classes[category_id-1]}: {score:.2f}',
                    color='red',
                    ha='left',
                    va='top',
                    fontsize=8
                )
                text.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), dpi=150, bbox_inches='tight', pad_inches=0)

        box_img = plt.imread(os.path.join(save_dir, f'{image_id}.jpg'))
        wandb.log({"inference_image": [wandb.Image(box_img, caption=f"{image_id}.jpg")]})

        plt.clf()

def test(device, image_size):

    wandb.init(project="wheat-detection",
               name= "test.py_result",
    )

    train_image_dir = 'data/global-wheat-detection/train'
    test_csv_path = 'data/global-wheat-detection/test_answer.csv'

    num_classes = 1

    test_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=get_transform(state='inference', image_size=image_size)
    )

    model = DetectionModel(num_classes=num_classes).make_model()
    model.load_state_dict(torch.load('wheat-faster-rcnn.pth'))
    model.to(device)

    visualize_predictions(test_data, device, model, 'examples/global-wheat-detection/faster-rcnn')
    print('Saved in ./examples/global-wheat-detection/faster-rcnn')


if __name__ == '__main__':
    test(args.device, args.image_size)