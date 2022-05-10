import pickle
import torch
import sys
sys.path.append("../model/convlstm/")

from training_seting import opt
from encoder_decoder import  EncoderDecoderConvLSTM
from moving_wavefront import MovingWFLightning
from wavefron_data_g import WaveFrontLoaderNorm
import numpy as np
import matplotlib.pyplot as plt
import cv2

SIZE = 64
def add_mask(img):
    width, height = SIZE,SIZE
    x, y = np.asarray([SIZE,SIZE]) // 2
    R = x.copy()
    X = np.arange(width).reshape(width, 1)
    Y = np.arange(height).reshape(1, height)
    mask_1 = ((X - x) ** 2 + (Y - y) ** 2) >= (R - 8) ** 2
    # mask_2 = ((X - x) ** 2 + (Y - y)**2) < (R-5)**2
    img[:,:,mask_1] = 0
    # img[mask_2] = 1
    return img


def build_unet_data(model_path, train_path = "",  data_upload_path  = ""):
    conv_lstm_model = EncoderDecoderConvLSTM(nf=opt.n_hidden_dim, in_chan=1).to("cuda")
    model = MovingWFLightning(model=
                              conv_lstm_model).to("cuda")

    model.load_state_dict(torch.load(model_path))
    model.eval()

    import gc
    # print(k)
    phases = []
    imgs1 = []
    imgs2 = []
    imgs3 = []
    coeffs = []

    for i in range(100):
        print(f"Loaded {i}/10 ...")
        import gc
        unet_data = None
        gc.collect()

       # print(gc.collect())
      #  print(gc.collect())
      #  print(gc.collect())
        unet_data = WaveFrontLoaderNorm(train_path, ep_len=25, trainp=0.1)

        with torch.no_grad():
          #  print("try detect")
            # train_data = WavefrontData(train_path, type_f = "PHASE")
            t = 0
            for batch in next(unet_data):
           #     print(len(batch))
              #  print(batch[2].shape)
               # raise
                train_batch = torch.from_numpy(batch[2]).float()
                loss = model.predict(train_batch.detach().to("cuda"), t)
                t+=1
              #  print('loss')
             #   print(batch[0].shape)
                photo = batch[0]
               # print(photo.shape)
             #   plt.imshow(photo[0][0].reshape(128,128))
             #   plt.colorbar()
             #   plt.show()
                pred = add_mask(loss["prediction"].detach().cpu().numpy())
                orig = add_mask(loss["original"].detach().cpu().numpy())

                residual = add_mask(orig - pred)

                img1 = photo[:, 20:, :]
                img2 = pred
                coeff = []

                phases.append(orig)
                imgs1.append(img1)
                imgs2.append(img2)
                imgs3.append(residual)
                coeffs.append(coeff)


    print("I am here!")
    import gc
    unet_data = None
    print(gc.collect())
  #  print(gc.collect())
    #print(gc.collect())
  #  print(gc.collect())
    print("Trying convert to numpy....")
    phases = np.asarray(phases)
    imgs1 = np.asarray(imgs1)
    imgs2 = np.asarray(imgs2)
    imgs3 = np.asarray(imgs3)
    coeffs = np.asarray(coeffs)

    arr_size = len(coeffs.ravel()) //(SIZE*SIZE)
    phases = phases.reshape(-1, SIZE, SIZE)
    imgs1 = imgs1.reshape(-1, SIZE, SIZE)
    imgs2 = imgs2.reshape(-1, SIZE, SIZE)
    imgs3 = imgs3.reshape(-1, SIZE, SIZE)
    coeffs = np.zeros(imgs3.shape[0])
    print(imgs1.shape)
    print("Example of data: ")
    plt.figure(figsize=(10, 10))
    for i in range(2):
        j = np.random.randint(0,3000)
        print(j)
        plt.subplot(2,4,1 + 4*i)
        plt.title("Sci-camera image")
        plt.imshow(imgs1[j])
        plt.colorbar(location = 'bottom', pad = 0.05)

        plt.subplot(2, 4, 2+ 4*i)
        plt.title("Conv-lstm prediction")
        plt.imshow(imgs2[j])
        plt.colorbar(location = 'bottom', pad = 0.05)

        plt.subplot(2, 4, 3+ 4*i)
        plt.title("Residual")
        plt.imshow(imgs3[j])
        plt.colorbar(location = 'bottom', pad = 0.05)

        plt.subplot(2, 4, 4 + 4*i)
        plt.title("Phase")
        plt.imshow(phases[j])
        plt.colorbar(location='bottom', pad=0.05)

    plt.tight_layout()
    plt.show()

    print("Saving...")
    import pickle
    with open(data_upload_path, 'wb') as f:
        pickle.dump((imgs1, imgs2, imgs3, phases, coeffs), f)


if __name__ == "__main__":
    #data = "/home/zoya/PycharmProjects/fswp/checkpoints/convlstm/CONV_LSTM_run_JordanRiddick_fd8d1d77-d634-4f6c-a150-fb6a5a94642e.pth"
   # data = "/home/zoya/PycharmProjects/fswp/checkpoints/convlstm/CONV_LSTM_run_AntonioFields_1f68f605-dcff-46a3-8b37-b838a5e06f17.pth"
    data = "/home/zoya/PycharmProjects/fswp/checkpoints/convlstm/CONV_LSTM_star_run_RobertAquilar_87d9194e-b083-4513-9421-d303c2630a50.pth"
    build_unet_data(data,
                    train_path="../data/data_pointstar_mixed_1_100x100.pickle", data_upload_path="../data/data_for_unet_star.pickle")
