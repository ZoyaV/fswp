import pickle
import torch
from torch import nn
from torch.nn.functional import mse_loss
import sys
import matplotlib.pyplot as plt
import argparse

sys.path.append("./model/convlstm/")
sys.path.append("./model/unet/")
sys.path.append("./model/")

from train_unet import UnetTrainer

print(" //// //// //// //// ")
print(torch.__version__)
print(torch.cuda.is_available())
print(" //// //// //// //// ")

def main(dir):
    convlstm_trainer = UnetTrainer(dir)
    convlstm_trainer.load_state_dict()
    preds_val, X_val_batch, y_val_batch, losses, fps = convlstm_trainer.predict_once()
    rmse = torch.sqrt(mse_loss(preds_val,y_val_batch))
    print("Images processed. FPS = %s"%fps)
    plt.subplot(1,2,1)
    plt.title(f"Predicted diffference ")

    plt.imshow(preds_val[0,0].cpu())
    plt.colorbar(location='bottom', pad=0.05)

    plt.subplot(1,2,2)
    plt.title("Original difference")
    plt.imshow(y_val_batch[0,0].cpu())
    plt.colorbar(location='bottom', pad=0.05)
    plt.tight_layout()
    plt.suptitle(f"Residuals prediction \n on batch RMSE = { round(float(rmse),3)}, total RMSE = {round(float(losses)**2,3)}"
                 f" \n FPS = {int(fps)}", fontsize=14)

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Unet example')
    parser.add_argument('config_dir', type=str, help='Input dir for config', default='../configs/run_solar_fswp.yaml')
    args = parser.parse_args()
   # print(args)
  #  print(args.config_dir)
    main(args.config_dir)