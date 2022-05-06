import pickle
import torch
from torch import nn
from torch.nn.functional import mse_loss
import sys
import matplotlib.pyplot as plt

sys.path.append("../model/convlstm/")
sys.path.append("../model/unet/")
sys.path.append("../model/")

from train_unet import UnetTrainer

print(" //// //// //// //// ")
print(torch.__version__)
print(torch.cuda.is_available())
print(" //// //// //// //// ")

if True:
    convlstm_trainer = UnetTrainer("../fswp_run.yaml")
    convlstm_trainer.load_state_dict()
    preds_val, X_val_batch, y_val_batch, losses, fps = convlstm_trainer.predict_once()
    rmse = torch.sqrt(mse_loss(preds_val,y_val_batch))
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