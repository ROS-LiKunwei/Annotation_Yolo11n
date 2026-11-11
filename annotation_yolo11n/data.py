"""Synchronized geometric augmentation in input pixels; coordinates remain semantic centers."""
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .schema import image_path


class PointDataset(Dataset):
    def __init__(self, rows, classes, images_root, input_size=(640, 480), augment=False):
        self.rows, self.classes = rows, classes
        self.width, self.height = input_size
        self.augment = augment
        if min(self.width, self.height) < 32 or self.width % 32 or self.height % 32:
            raise ValueError("Input dimensions must be divisible by32")
        self.images = []
        for row in rows:
            image = cv2.imread(str(image_path(images_root, row)))
            if image is None:
                raise ValueError(f"Cannot load {row['id']}")
            if image.shape[:2] != (row["height"], row["width"]):
                raise ValueError(f"Image dimensions differ: {row['id']}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.shape[:2] != (self.height, self.width):
                image = cv2.resize(image, (self.width, self.height))
            self.images.append(image)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row, image = self.rows[index], self.images[index].copy()
        points = np.zeros((len(self.classes), 2), np.float32)
        status = np.zeros(len(self.classes), np.int64)
        for j, name in enumerate(self.classes):
            obj = row["objects"][name]
            status[j] = 2 if obj["status"] == "visible" else 1 if obj["status"] in {"ignore", "pending"} else 0
            if status[j] == 2:
                points[j] = np.asarray(obj["point_xy"]) * [self.width / row["width"], self.height / row["height"]]
        if self.augment:
            if random.random() < .5:
                image = image[:, ::-1].copy()
                points[:, 0] = self.width - 1 - points[:, 0]
            if random.random() < .75:
                scale = random.uniform(.9, 1.1)
                dx, dy = random.uniform(-.05, .05) * self.width, random.uniform(-.05, .05) * self.height
                matrix = np.array([[scale, 0, (1 - scale) * self.width / 2 + dx],
                                   [0, scale, (1 - scale) * self.height / 2 + dy]], np.float32)
                image = cv2.warpAffine(image, matrix, (self.width, self.height), borderValue=(114, 114, 114))
                points = points @ matrix[:, :2].T + matrix[:, 2]
                lost = ((status == 2) & ((points[:, 0] < 0) | (points[:, 0] >= self.width)
                                        | (points[:, 1] < 0) | (points[:, 1] >= self.height)))
                status[lost] = 0
            image = np.clip(image.astype(np.float32) * random.uniform(.85, 1.15) + random.uniform(-8, 8), 0, 255).astype(np.uint8)
        return (torch.from_numpy(image.transpose(2, 0, 1).copy()).float() / 255,
                torch.from_numpy(points / 4), torch.from_numpy(status))
