import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse

import sys
sys.path.append("./model/convlstm/")
sys.path.append("./model/unet/")
sys.path.append("./model/")
sys.path.append("./ao_env")

import gym
from ao_env import AdaptiveOptics
from wrappers import FrameStack

from training_seting import opt
from encoder_decoder import  EncoderDecoderConvLSTM
from moving_wavefront import MovingWFLightning
from wavefron_data_g import WaveFrontLoaderNorm
from train_unet import UnetTrainer


SIZE = 64
def add_mask(img, f = 0):
    width, height = SIZE,SIZE
    x, y = np.asarray([SIZE,SIZE]) // 2
    R = x.copy()
    X = np.arange(width).reshape(width, 1)
    Y = np.arange(height).reshape(1, height)
    mask_1 = ((X - x) ** 2 + (Y - y) ** 2) >= (R - 8) ** 2
    # mask_2 = ((X - x) ** 2 + (Y - y)**2) < (R-5)**2
    img[mask_1] = f
    # img[mask_2] = 1
    return img

def bnorm(d):
    mins = d.min(axis=(1,2,3,4))
    maxs = d.max(axis=(1,2,3,4))
    d = (d - mins.reshape(-1,1,1,1,1))/(maxs.reshape(-1,1,1,1,1) -mins.reshape(-1,1,1,1,1))
    return d, mins, maxs

def main(args):

    unet_trainer = UnetTrainer(args['config_dir'])
    unet_trainer.load_state_dict()

    model_path = args['model_path']

    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1).to("cuda")

    model = MovingWFLightning(model=
                              conv_lstm_model).to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()


    env = AdaptiveOptics("ao_env/sh_8x8.yaml")
    env = FrameStack(env)
    obs = env.reset()
    r = 0
    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 900, 900)
    for i in range(300):

        if False:
            plt.imshow(obs[-1,0]**0.5)
            plt.title("Step - %d; IntensityDistribution %.2f"%(i+23, r))
            plt.savefig(f"{i}.png")
            plt.colorbar()
            plt.show()

        cv2.imshow('image', obs[-1,0]**0.5)
        cv2.waitKey(1)


        d = obs[:,1].reshape(1,20,64,64,1)
        d,mins, maxs = bnorm(d)

        d = torch.from_numpy(d).to(torch.float)
        a = model.predict(d.detach().to("cuda"), 0)['prediction']
        a = a.cpu().detach().numpy()
        conv_pred = a.copy()

        obs_in,_,_ = bnorm(obs[:,0].reshape(1,20,64,64,1))
        inp = (a[0] -a[0].min())/(a[0].max() - a[0].min())
        inp = add_mask(inp)

        unet_input = np.asarray([obs_in[0,-1,:,:,0], inp])
        unet_input = torch.from_numpy(unet_input)
        unet_input = unet_input.reshape(1,2,64,64).to(torch.float)
        residual = unet_trainer.model(unet_input)
        original = d[0, -1, :, :, 0].cpu().detach().numpy()

        a_ = add_mask(a[0], f = original[0,0])
        a_ = a_ * (maxs - mins) + (mins)
        res =  residual[0,0].cpu().detach().numpy()
        preds =  a[0] + res
        preds_ =  a[0] - res
        original = original * (maxs - mins) + (mins)
        preds = preds * (original - mins) + (mins)

        preds = add_mask(preds,f = original[0,0])
        preds_ = add_mask(preds_,f = original[0,0])

        obs, r, _, _ = env.step(a_)
       # obs, r, _, _ = env.step(np.ones_like(a_))


   # plt.show()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run pipeline')
    parser.add_argument('config_dir', type=str, help='Input dir for config', default="../configs/run_star_fswp.yaml")
    parser.add_argument('savefig', type=bool, default=False)
    parser.add_argument('model_path', type=str,
                        default="/home/zoya/PycharmProjects/fswp/checkpoints/convlstm/CONV_LSTM_star_run_RobertAquilar_87d9194e-b083-4513-9421-d303c2630a50.pth")
    args = parser.parse_args()
    main(vars(args))
