import argparse
import os
import random
import shutil
import wandb

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe

import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision

from src.dataset import WheatDataset, collate_fn, get_transform
from src.model import DetectionModel
from src.utils import split_dataset, MeanAveragePrecision, check_train_batch_size


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="cpu", type=str, help="device to use for training")
parser.add_argument("--batch_size", default=16, type=int, help="number of samples in a batch")
parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train the model")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for the optimizer")
parser.add_argument("--image_size", default=1024, type=int, help="int for image resize")
args = parser.parse_args()


def denormalize_image(image):
    mean = np.array((0.485, 0.456, 0.406))
    std = np.array((0.229, 0.224, 0.225))
    image = image * std.reshape((1, 3, 1, 1)) + mean.reshape((1, 3, 1, 1))
    image = image * 255.0
    image = np.clip(image, 0, 1)
    return image


def visualize_dataset(image_dir: os.PathLike, csv_path: os.PathLike, image_size: int, state: str, save_dir: os.PathLike, n_images: int = 10) -> None:

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    dataset = WheatDataset(
        image_dir=image_dir,
        csv_path=csv_path,
        transform=get_transform(state=state, image_size=image_size)
    )

    indexes = random.choices(range(len(dataset)), k=n_images)
    for i in indexes:
        image, target, image_id = dataset[i]
        if state == 'train':
            image = denormalize_image(image)[0]
        image = image.numpy().transpose(1, 2, 0)

        plt.imshow(image)
        ax = plt.gca()

        for x1, y1, x2, y2 in target['boxes']:
            w = x2 - x1
            h = y2 - y1

            category_id = 'wheat'

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
                category_id,
                color='red',
                ha='left',
                va='top',
                fontsize=8
            )
            text.set_path_effects([pe.withStroke(linewidth=3, foreground='black')])

        plt.axis('off')
        plt.savefig(os.path.join(save_dir, f'{image_id}.jpg'), bbox_inches='tight', pad_inches=0)
        plt.close()


def train_one_epoch(
    dataloader: DataLoader,
    device: str,
    model: torchvision.models.detection,
    scheduler: torch.optim.lr_scheduler.OneCycleLR,
    optimizer: torch.optim.Optimizer
    ) -> None:

    size = len(dataloader.dataset)
    model.train()

    scaler = torch.cuda.amp.GradScaler()
    for batch, (images, targets, image_ids) in enumerate(dataloader):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.cuda.amp.autocast():
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()    
        scale_after = scaler.get_scale()
        if scale_before <= scale_after:
                scheduler.step()
        if batch % 10 == 0:
            if batch == 0:
                wandb.log({
                    "noramlized_train_batch": [wandb.Image(images[i], caption=f"{image_ids[i]}.jpg") for i in range(len(images))],
                    "origin_train_batch": [wandb.Image(denormalize_image(images[i].cpu()), caption=f"{image_ids[i]}.jpg") for i in range(len(images))],
                })
            wandb.log({
                "total_loss": loss.item(),
                "loss_classifier": loss_dict['loss_classifier'].item(),
                "loss_box_reg": loss_dict['loss_box_reg'].item(),
                "loss_objectness": loss_dict['loss_objectness'].item(),
                "loss_rpn_box_reg": loss_dict['loss_rpn_box_reg'].item()
            })

            current = batch * len(images)
            message = 'total loss: {:.4f}, cls loss: {:>4f}, box loss: {:>4f}, obj loss: {:>4f}, rpn loss: {:>4f}  [{:>5d}/{:>5d}]'
            message = message.format(
                loss.item(),
                loss_dict['loss_classifier'].item(),
                loss_dict['loss_box_reg'].item(),
                loss_dict['loss_objectness'].item(),
                loss_dict['loss_rpn_box_reg'].item(),
                current,
                size
            )
            print(message)
    print()


def val_one_epoch(
    dataloader: DataLoader,
    device: str,
    model: torchvision.models.detection,
    image_size: int,
    metric) -> None:
 
    num_batches = len(dataloader)
    test_loss = 0
    test_cls_loss = 0
    test_box_loss = 0
    test_obj_loss = 0
    test_rpn_loss = 0
    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
            with torch.cuda.amp.autocast():
                model.train()
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())

                test_loss += loss
                test_cls_loss += loss_dict['loss_classifier']
                test_box_loss += loss_dict['loss_box_reg']
                test_obj_loss += loss_dict['loss_objectness']
                test_rpn_loss += loss_dict['loss_rpn_box_reg']

                model.eval()
                preds = model(images)

            metric.update(preds, image_ids, image_size)
    test_loss /= num_batches
    test_cls_loss /= num_batches
    test_box_loss /= num_batches
    test_obj_loss /= num_batches
    test_rpn_loss /= num_batches
    iou_score = metric.compute()

    print(f'\nTest Error: \n Avg loss: {test_loss:>8f} \n Class loss: {test_cls_loss:>8f} \n Box loss: {test_box_loss:>8f} \n Obj loss: {test_obj_loss:>8f} \n RPN loss: {test_rpn_loss:>8f} \n')
    wandb.log({
        "AP [ IoU=0.50:0.95 | area=all | maxDets=100 ]": iou_score[0],
        "AP [ IoU=0.50      | area=all | maxDets=100 ]": iou_score[1],
        "AP [ IoU=0.75      | area=all | maxDets=100 ]": iou_score[2],
        "AP [ IoU=0.50:0.95 | area=  s | maxDets=100 ]": iou_score[3],
        "AP [ IoU=0.50:0.95 | area=  m | maxDets=100 ]": iou_score[4],
        "AP [ IoU=0.50:0.95 | area=  l | maxDets=100 ]": iou_score[5],
        "AR [ IoU=0.50:0.95 | area=all | maxDets=  1 ]": iou_score[6],
        "AR [ IoU=0.50:0.95 | area=all | maxDets= 10 ]": iou_score[7],
        "AR [ IoU=0.50:0.95 | area=all | maxDets=100 ]": iou_score[8],
        "AR [ IoU=0.50:0.95 | area=  s | maxDets=100 ]": iou_score[9],
        "AR [ IoU=0.50:0.95 | area=  m | maxDets=100 ]": iou_score[10],
        "AR [ IoU=0.50:0.95 | area=  l | maxDets=100 ]": iou_score[11],
    })
    wandb.log({
        "inference_batch": [wandb.Image(images[i].cpu(), caption=f"{image_ids[i]}.jpg") for i in range(len(images))],
        "test_loss": test_loss,
        "test_cls_loss": test_cls_loss,
        "test_box_loss": test_box_loss,
        "test_obj_loss": test_obj_loss,
        "test_rpn_loss": test_rpn_loss
    })

    metric.reset()
    print()


def train(
    model: torchvision.models.detection,
    device: str = 'cpu', 
    batch_size: int = 16,
    epochs: int = 10,
    lr: float = 1e-3,
    image_size: int = 1024
    ) -> None:
   

    wandb.init(project="wheat-detection",
               name= "train.py_result",
               config={
                    "batch_size": batch_size,
                    "epochs": epochs,
                    "learning_rate": lr,
                    "device": device
                }
    )

    csv_path = 'data/global-wheat-detection/train.csv'
    train_image_dir = 'data/global-wheat-detection/train'
    train_csv_path = 'data/global-wheat-detection/train_answer.csv'
    test_csv_path = 'data/global-wheat-detection/test_answer.csv'
    
    num_classes = 1
    split_dataset(csv_path)
    visualize_dataset(train_image_dir, train_csv_path, image_size, 'train', save_dir='examples/global-wheat-detection/train')
    visualize_dataset(train_image_dir, test_csv_path, image_size, 'test', save_dir='examples/global-wheat-detection/test')

    training_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=train_csv_path,
        transform=get_transform(state='train', image_size=image_size),
    )
    test_data = WheatDataset(
        image_dir=train_image_dir,
        csv_path=test_csv_path,
        transform=get_transform(state='test', image_size=image_size),
    )

    model = DetectionModel(num_classes=num_classes).make_model().to(device)

    
    batch_size = check_train_batch_size(model=model, imgsz=image_size)

    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, batch_size=int(batch_size/4), num_workers=0, collate_fn=collate_fn)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.005)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_dataloader), epochs=epochs)
    metric = MeanAveragePrecision(csv_path=csv_path)

    for t in range(epochs):
        print(f'Epoch {t+1}\n-------------------------------')
        train_one_epoch(train_dataloader, device, model, scheduler, optimizer)
        val_one_epoch(test_dataloader, device, model, image_size, metric)
    print('Done!')

    torch.save(model.state_dict(), 'wheat-faster-rcnn.pth')
    print('Saved PyTorch Model State to wheat-faster-rcnn.pth')


if __name__ == '__main__':
    train(model=DetectionModel, device=args.device, batch_size=args.batch_size, \
          epochs=args.epochs, lr=args.lr, image_size=args.image_size)