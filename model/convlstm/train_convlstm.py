
import torch
from training_seting import opt
from encoder_decoder import  EncoderDecoderConvLSTM
from moving_wavefront import MovingWFLightning
from wavefron_data_g import WaveFrontLoaderNorm

import sys
sys.path.append("../")
from base import BaseTrainer


class ConvLstmTrainer(BaseTrainer):
    def __init__(self, args):
        super(ConvLstmTrainer, self).__init__(args)
        self.model_base = "CONV_LSTM"
        self.load_configs()
        self.init_model()
        self.load_dataset()

    def init_model(self):
        device = self.args["device"]["to"]
        conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1).to(device)
        self.model = MovingWFLightning(model=conv_lstm_model).to(device)
        self.lr = [float(lr) for lr in self.args["convlstm"]["lr"]]
        self.change_lr = self.args["convlstm"]["change_lr"]
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr[0])
        return

    def load_configs(self):
        self.args = super().load_configs()
        self.save_path = self.args['convlstm']['checkpoints_path']

    def load_dataset(self):
        self.train_data = WaveFrontLoaderNorm(self.args['convlstm']['train_path'], ep_len=25, trainp=0.8)
        self.val_data = WaveFrontLoaderNorm(self.args['convlstm']['val_path'], ep_len=25, trainp=0.1)
        self.test_data = WaveFrontLoaderNorm(self.args['convlstm']['test_path'], ep_len=25, trainp=0.1)

    def train_model(self):
        last_epoch = 0
        min_loss = 100
        for epoch in range(self.args['convlstm']['epoches']):
            j = 0
            losses = 0
            tlosses = 0
            jtest = 0
            i = 0
            for batch in next(self.train_data):
                self.optimizer.zero_grad()
                j += 1
                #    print(batch[2].shape)
                train_batch = torch.from_numpy(batch[2]).float()
                loss = self.model.training_step(train_batch.to("cuda"), i)
                losses += loss["loss"]
                self.optimizer.step()
                i += 1
            k = 0
            for tbatch in next(self.test_data):
                with torch.no_grad():
                    test_batch = torch.from_numpy(tbatch[2]).float()
                    test_loss = self.model.test_step(test_batch.detach().to("cuda"), k)
                    jtest += 1
                    tlosses += test_loss["loss"]
                    k += 1
            if epoch<self.change_lr[-1] and epoch == self.change_lr[last_epoch]:
                print(f"I am in {self.change_lr[last_epoch]}")
                last_epoch += 1
                print(f"Changing lr from {self.lr[last_epoch-1]} to {self.lr[last_epoch]}")
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr[last_epoch])
                try:
                    print(f"Wait {self.change_lr[last_epoch]}")
                except:
                    print("Wait end of train")

            print(f"Epoch {epoch}- Loss:{round((float(losses) / j), 2)} TestLoss:{round((float(tlosses) / k), 2)}")
            mean_loss = (losses/j) + (tlosses/ k)
            mean_loss = mean_loss/2
            if mean_loss < min_loss:
                min_loss = mean_loss
                path = self.buity_name()
                torch.save(self.model.state_dict(), path)
                print()
                print("New best model! Save in ->")
                print(path)
                print()


        return path


if __name__ == "__main__":
    convlstm_trainer = ConvLstmTrainer("../../fswp_train.yaml")
    convlstm_trainer.train_model()



