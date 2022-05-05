import torch.nn as nn
import torch
import os
import torchvision
import pytorch_lightning as pl

# from google.colab import drive
# drive.mount('/content/gdrive')

# import sys
# import os
# py_file_location = '/content/gdrive/My Drive/ao_prediction/notebooks'
# sys.path.append(os.path.abspath(py_file_location))

from training_seting import opt
from conv_lstm_cell import ConvLSTMCell
from encoder_decoder import  EncoderDecoderConvLSTM

class MovingWFLightning(pl.LightningModule):

    def __init__(self, hparams=None, model=None):
        super(MovingWFLightning, self).__init__()

        # default config
        self.path = os.getcwd() + '/data'
        self.model = model

        # logging config
        self.log_images = True

        # Training config
        self.criterion = torch.nn.MSELoss()
        self.batch_size = opt.batch_size
        self.n_steps_past = opt.ahead
        self.n_steps_ahead = opt.predict  # 4

    def create_video(self, x, y_hat, y):
        # predictions with input for illustration purposes
        preds = torch.cat([x.cpu(), y_hat.unsqueeze(2).cpu()], dim=1)[0]

        # entire input and ground truth
        y_plot = torch.cat([x.cpu(), y.unsqueeze(2).cpu()], dim=1)[0]

        # error (l2 norm) plot between pred and ground truth
        difference = (torch.pow(y_hat[0] - y[0], 2)).detach().cpu()
        zeros = torch.zeros(difference.shape)
        difference_plot = torch.cat([zeros.cpu().unsqueeze(0), difference.unsqueeze(0).cpu()], dim=1)[
            0].unsqueeze(1)

        # concat all images
        final_image = torch.cat([preds, y_plot, difference_plot], dim=0)

        # make them into a single grid image file
        grid = torchvision.utils.make_grid(final_image, nrow=self.n_steps_past + self.n_steps_ahead)

        return grid

    def forward(self, x):
        x = x.to(device='cuda')

        output = self.model(x, future_seq=self.n_steps_ahead)

        return output

    def criterion(self, y_hat, y):
        loss_f = nn.MSELoss()
        return  torch.sqrt(torch.sqrt(loss_f(y_hat, y)))
        
    def training_step(self, batch, batch_idx):
        #   self.zero_grad()
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?
        loss = self.criterion(y_hat, y)
        loss.backward()
        # self.zero_grad()

        return {'loss': loss, "prediction": y_hat, "original": y}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        #  self.zero_grad()
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?
        loss = self.criterion(y_hat, y)
        # self.zero_grad()
        return {'loss': loss, "prediction": y_hat, "original": y}
    
    def predict(self, batch, batch_idx):
        # OPTIONAL
        #  self.zero_grad()
        x, y = batch[:, 0:self.n_steps_past, :, :, :], batch[:, self.n_steps_past:, :, :, :]
        x = x.permute(0, 1, 4, 2, 3)
        y = y.squeeze()
        y_hat = self.forward(x).squeeze()  # is squeeze neccessary?
        #loss = self.criterion(y_hat, y)
        # self.zero_grad()
        return { "prediction": y_hat, "original": y}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=opt.lr)

    # @pl.data_loader
    def train_dataloader(self):
        data_generator = WavefrontData(path, test=False)
        return data_generator

    # @pl.data_loader
    def test_dataloader(self):
        data_generator = WavefrontData(path, test=True)
        return data_generator
