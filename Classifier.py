import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report
from torch.optim import lr_scheduler
from torchvision.models import resnet50, ResNet50_Weights
from tqdm import tqdm


class Classifier:
    def __init__(self, dataMananger, load_model=True, filename=None):
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
            self.load_model(filename=filename)

        self.model.to(self.device)
        self.create_loss_function()

    def create_loss_function(self):
        # def custom_loss(y_pred_logits, y_true):
        #     """ Do what you want here, then return the loss """
        #     loss = None
        #     return loss
        # self.loss_function = custom_loss
        # self.loss_function = nn.CrossEntropyLoss()

        from CostSensitiveLoss import CostSensitiveLoss
        # 定义成本矩阵
        cost_matrix = [
            [0, 2, 5, 3, 4],
            [2, 0, 6, 4, 3],
            [5, 6, 0, 1, 2],
            [3, 4, 1, 0, 5],
            [4, 3, 2, 5, 0]
        ]
        # 使用自定义损失函数
        self.loss_function = CostSensitiveLoss(cost_matrix)

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
                torch.load(filename)
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

    def test(self, on_train_set=False, file=None):
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
        print("Confusion matrix: \n", M, file=file)
        print(classification_report(y_true_all, y_pred_all), file=file)
