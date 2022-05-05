
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from base import BaseTrainer
from in_preprocessing import ImageProcessor
from torch.nn.functional import normalize
import datetime

class UnetTrainer(BaseTrainer):
    def __init__(self, args):
        super(UnetTrainer, self).__init__(args)
        self.model_base = "UNET"
        self.save_path = "."
        self.load_configs()
        self.img_processor = ImageProcessor(self.args['image']['size_unet'], self.args['image']['dataset_lenghtn'])
        self.init_model()
        self.load_dataset()

    def init_model(self):
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                               in_channels=3, out_channels=1, init_features=64, pretrained=False)

        in_block =  nn.Sequential(nn.Conv2d(2, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                      nn.BatchNorm2d(3, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                      nn.ReLU(inplace=True))

        out_block =   nn.Sequential(nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True), nn.Tanh())



        self.model = nn.Sequential(in_block, model,out_block)
        self.model.to(self.args["device"]["to"])
        self.loss_f = nn.MSELoss()
        self.lr = [float(lr) for lr in self.args["unet"]["lr"]]
        self.change_lr = self.args["unet"]["change_lr"]
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr[0])
        return

    def load_dataset(self):
        self.X_train, _, self.y_train, self.X_val, _, self.y_val = self.img_processor.get_traintest_data(self.args['unet']['train_path'],
                                                                              outputs=2, img_deap=1)

        print(self.X_train.shape)
        print(self.y_train.shape)
        if True:
            plt.subplot(131)
            plt.imshow(self.X_train[0][0])
            plt.colorbar(location='bottom', pad=0.05)
            plt.subplot(132)
            plt.imshow(self.X_train[0][1])
            plt.colorbar(location='bottom', pad=0.05)
            plt.subplot(133)
            plt.imshow(self.y_train[0][0])
            plt.title(self.y_train.shape)
            plt.colorbar(location='bottom', pad=0.05)
            plt.show()


       # plt.imshow(self.X_train[0])
        return

    def load_configs(self):
        self.args = super().load_configs()
        self.save_path = self.args['unet']['checkpoints_path']
       # print("ALOOOO!!!!!")

    def load_state_dict(self):
        path = self.args['unet']['wights_path']
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        pass

    def predict_once(self):
        with torch.no_grad():
            val_batch_size = self.args["unet"]["val_batch_size"]
            order = np.random.permutation(len(self.X_val))
            val_loss = 0.
            count = len(self.X_val)/val_batch_size
            fps = len(self.X_val)
            a = datetime.datetime.now()
            for start_index in range(0, len(self.X_val), val_batch_size):
                self.optim.zero_grad()
                batch_indexes = order[start_index:start_index + val_batch_size]
                X_val_batch = self.X_val[batch_indexes].to(self.args["device"]["to"])
                y_val_batch = self.y_val[batch_indexes].to(self.args["device"]["to"])

                preds_val = self.model.forward(X_val_batch.detach())

                loss_value_val = torch.sqrt(torch.sqrt(self.loss_f(preds_val.detach(), y_val_batch.detach())))
                val_loss += loss_value_val
            b = datetime.datetime.now()
            delta = b - a
            fps = len(self.X_val) / delta.total_seconds()
            return preds_val, X_val_batch, y_val_batch, val_loss/count, fps

    def train_model(self):
        it = 0
        min_loss = np.inf
        last_epoch = 0
        batch_size = self.args["unet"]["batch_size"]
        val_batch_size = self.args["unet"]["val_batch_size"]
        self.model.train()
        min_loss = 100
        device = self.args["device"]["to"]
        for epoch in range(self.args["unet"]["epoches"]):
            order = np.random.permutation(len(self.X_train))
            loss = 0.
            for start_index in range(0, len(self.X_train), batch_size):
                self.optim.zero_grad()
                batch_indexes = order[start_index:start_index + batch_size]
                X_batch = self.X_train[batch_indexes].to(device)
                y_batch = self.y_train[batch_indexes].to(device)
                if False:
                    plt.subplot(131)
                    plt.imshow(X_batch[0][0].cpu())
                    plt.colorbar(location='bottom', pad=0.05)
                    plt.subplot(132)
                    plt.imshow(X_batch[0][1].cpu())
                    plt.colorbar(location='bottom', pad=0.05)
                    plt.subplot(133)
                    plt.imshow(y_batch[0][0].cpu())
                    plt.colorbar(location='bottom', pad=0.05)
                    plt.show()
                preds = self.model.forward(X_batch)

              #  preds = (preds - mins.reshape(-1, 1, 1, 1)) / (
               #             maxs.reshape(-1, 1, 1, 1) - mins.reshape(-1, 1, 1, 1))
              #  y_batch = (preds - mins.reshape(-1, 1, 1, 1)) / (
                  #      maxs.reshape(-1, 1, 1, 1) - mins.reshape(-1, 1, 1, 1))
               # preds = normalize(preds)
               # y_batch = normalize(y_batch)
                loss_value = torch.sqrt(torch.sqrt(self.loss_f(preds, y_batch)))
                loss += loss_value
                loss_value.backward()

                self.optim.step()
            if epoch % 2 == 0:
                order = np.random.permutation(len(self.X_val))
                val_loss = 0.
                for start_index in range(0, len(self.X_val), val_batch_size):
                    self.optim.zero_grad()
                    batch_indexes = order[start_index:start_index + val_batch_size]
                    X_val_batch = self.X_val[batch_indexes].to(device)
                    y_val_batch = self.y_val[batch_indexes].to(device)

                    preds_val = self.model.forward(X_val_batch.detach())

                    loss_value_val = torch.sqrt(torch.sqrt(self.loss_f(preds_val.detach(), y_val_batch.detach())))
                    val_loss += loss_value_val
            if epoch % 2 == 0:
                plt.subplot(1, 2, 1)
                plt.imshow(preds[0, 0].cpu().detach().numpy())
                plt.colorbar(location='bottom', pad=0.05)
                plt.subplot(1, 2, 2)
                plt.imshow(y_batch[0, 0].cpu().detach().numpy())
                plt.colorbar(location='bottom', pad=0.05)
                plt.show()
            coef_train = len(self.X_train) / batch_size
            coef_val = len(self.X_val) / val_batch_size
            #train_losses.append(loss / coef_train)
           # val_losses.append(val_loss / coef_val)
            mean_loss = (loss / coef_train) + (val_loss / coef_val)
            mean_loss = mean_loss / 2
            if mean_loss < min_loss:
                min_loss = mean_loss
                path = self.buity_name()
                torch.save(self.model.state_dict(), path)
                print()
                print("New best model! Save in ->")
                print(path)
                print()

            print("EPOCH - %d, train loss - %f, val loss - %f" % (epoch, loss / coef_train, val_loss / coef_val))
            if epoch < self.change_lr[-1] and epoch == self.change_lr[last_epoch]:
                print(f"I am in {self.change_lr[last_epoch]}")
                last_epoch += 1
                print(f"Changing lr from {self.lr[last_epoch - 1]} to {self.lr[last_epoch]}")
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr[last_epoch])
                try:
                    print(f"Wait {self.change_lr[last_epoch]}")
                except:
                    print("Wait end of train")
        self.model.eval()



if __name__ == "__main__":
    convlstm_trainer = UnetTrainer("../../fswp_train.yaml")
    convlstm_trainer.train_model()
