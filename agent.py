import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim import lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm

### Exact same code as Day 2 ###
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
root = "."


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


data_dir = 'data'

dataset = DataManager(data_dir)
dataset.show_sample_images()

images_bchw, y_true = dataset.get_sample_model_input()
print(type(images_bchw), images_bchw.shape)
print(type(y_true), y_true.shape)


class Classifier:
    def __init__(self, dataMananger, load_model=True):
        self.dataMananger = dataMananger
        self.trainloader = self.dataMananger.trainloader
        self.testloader = self.dataMananger.testloader
        self.classes = self.dataMananger.classes
        self.artifacts_dir = "./artifact/"
        if not os.path.exists(self.artifacts_dir):
            os.makedirs(self.artifacts_dir)

        self.device = torch.device(
            torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
        )
        print("Using device %s" % self.device)

        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.replace_model_last_layer(len(self.classes))

        if load_model is True:
            self.load_model()

        self.model.to(self.device)
        self.create_loss_function()

    def create_loss_function(self):
        def custom_loss(y_pred_logits, y_true):
            """ Do what you want here, then return the loss """
            loss = None
            return loss

        # self.loss_function = custom_loss
        self.loss_function = nn.CrossEntropyLoss()

    def replace_model_last_layer(self, n_class):
        last_layer_name, last_layer = list(self.model.named_modules())[-1]

        featveclen = last_layer.weight.shape[1]
        exec("self.model.%s = nn.Linear(%s,%s)" % \
             (last_layer_name, featveclen, n_class)
             )

    def save_model(self):
        if not os.path.exists(self.artifacts_dir + "checkpoint/"):
            os.makedirs(self.artifacts_dir + "checkpoint/")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        torch.save(
            self.model.state_dict(),
            self.artifacts_dir + "checkpoint/model_%s.pt" % timestamp
        )
        print("model_%s.pt successfully saved!" % timestamp)
        return timestamp

    def load_model(self, filename=None):
        try:
            if filename is None:
                # get latest file, since files are named by date_time
                filename = sorted(
                    os.listdir(self.artifacts_dir + "checkpoint/")
                )[-1]
            self.model.load_state_dict(
                torch.load("%s/checkpoint/%s" % (self.artifacts_dir, filename))
            )
            print("%s successfully loaded..." % filename)
        except:
            print("Unable to load %s..." % filename)

    def train(self, epochs=1, lr=1e-3, save=True):
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = \
            lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=100, T_mult=1)

        self.history = []
        print("Beginning training for %d epochs" % epochs)
        print("lr: ", self.optimizer.param_groups[0]['lr'])

        self.model.train()
        for epoch in range(epochs):
            print("Epoch %d" % (epoch + 1))
            for i, data in tqdm(enumerate(self.trainloader)):
                images, y_true = data
                images, y_true = images.to(self.device), y_true.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_function(outputs, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                self.scheduler.step()

                if (i + 1) % 10 == 0:
                    self.history.append(loss.item())
                    print("Epoch %d Batch %d -- loss: %.3f" % (epoch + 1, i + 1, loss.item()))

            if save is True:
                self.save_model()

    def test(self, on_train_set=False):
        holder = {}
        holder['y_true'] = []
        holder['y_hat'] = []

        if on_train_set is True:
            print("Predicting on train set to get metrics")
            dataloader = self.trainloader
        else:
            print("Predicting on eval set to get metrics")
            dataloader = self.testloader

        self.model.eval()
        with torch.no_grad():
            for data in dataloader:
                images, y_true = data
                images, y_true = images.to(self.device), y_true.to(self.device)

                outputs = self.model(images)
                _, y_hat = torch.max(outputs, 1)  # logits not required, index pos is sufficient
                holder['y_true'].extend(
                    list(y_true.cpu().detach().numpy())
                )
                holder['y_hat'].extend(
                    list(y_hat.cpu().detach().numpy())
                )

        y_true_all = holder['y_true']
        y_pred_all = holder['y_hat']
        M = confusion_matrix(y_true_all, y_pred_all)
        print("Confusion matrix: \n", M)
        print(classification_report(y_true_all, y_pred_all))


classifier = Classifier(
    dataset, load_model=False
)

classifier.train(lr=1e-4, epochs=5)

classifier.test(on_train_set=True)
classifier.test(on_train_set=False)
