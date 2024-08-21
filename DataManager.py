import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class DataManager:
    def __init__(self, data_dir):
        self.C, self.H, self.W = 3, 224, 224
        self.batch_size = 8
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(self.H, self.W)),
            transforms.Normalize(mean, std)
        ])
        self.transform_ag = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.10, contrast=0.10, saturation=0.10, hue=0.10
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(
                size=(self.H, self.W), scale=(0.90, 1.00), ratio=(0.90, 1.10)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        self.trainset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transform=self.transform_ag
        )
        self.testset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, 'test'),
            transform=self.transform
        )
        self.class_to_idx = self.trainset.class_to_idx
        self.classes = list(self.class_to_idx.keys())

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size,
            shuffle=True, num_workers=0
        )
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size,
            shuffle=False, num_workers=0
        )

    def show_sample_images(self):
        data = next(iter(self.trainloader))
        images_bchw, y_true = data

        fig, axes = plt.subplots(2, 2)
        images_bhwc = np.transpose(
            images_bchw.numpy(), (0, 2, 3, 1)
        )
        axes[0, 0].imshow(images_bhwc[0,])
        axes[0, 0].set_title("%s" % self.classes[y_true[0]])
        axes[0, 1].imshow(images_bhwc[1,])
        axes[0, 1].set_title("%s" % self.classes[y_true[1]])
        axes[1, 0].imshow(images_bhwc[2,])
        axes[1, 0].set_title("%s" % self.classes[y_true[2]])
        axes[1, 1].imshow(images_bhwc[3,])
        axes[1, 1].set_title("%s" % self.classes[y_true[3]])
        plt.tight_layout()
        plt.show()

    def get_sample_model_input(self):
        data = next(iter(self.trainloader))
        images_bchw, y_true = data

        return images_bchw, y_true