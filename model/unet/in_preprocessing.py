import numpy as np
import cv2
import pickle
import torch


def renormalize(img, min_v, max_v):
    return (img * (max_v - min_v)) + (min_v)

def batch_normalize(imgs, safe_parametrs=False):
    if not safe_parametrs:
        return (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)), None, None
    else:
        return (imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)), np.min(imgs), np.max(imgs)

def point_normalize(imgs, safe_parametrs=False):
    new_imgs = []
    minv, maxv = [], []
    for img in imgs:
        # print(img.shape)
        if (len(img.shape) >= 3) and (img.shape[-1] > 1):
            new_img_1, _, _ = batch_normalize(img[0, :, :], safe_parametrs)
            new_img_2, _, _ = batch_normalize(img[1, :, :], safe_parametrs)
            new_img = np.append(new_img_1[np.newaxis, :, :], new_img_2[np.newaxis, :, :], axis=0)
            try:
                new_img_3, _, _ = batch_normalize(img[2, :, :], safe_parametrs)
                new_img = np.append(new_img, new_img_3[np.newaxis, :, :], axis=0)
            except:
                pass
        else:
            if safe_parametrs:
                new_img, min_val, max_val = batch_normalize(img, safe_parametrs)
                minv.append(min_val)
                maxv.append(max_val)
            new_img, _, _ = batch_normalize(img, safe_parametrs)
        new_imgs.append(new_img)
    # print(np.asarray(new_imgs).shape)
    return np.asarray(new_imgs), minv, maxv

def filter_good_idx(X):
    # отбираем только с хорошим индексом
    good_indx = []
    for i, x in enumerate(X):
        if np.sum(np.isnan(x)) == 0:
            good_indx.append(i)
    return good_indx


class ImageProcessor():
    def __init__(self, img_size, img_num):
        self.IMG_SIZE = img_size
        self.IMG_NUM = img_num

    def preprocessing(self, X, Y, Y_sumphase, outputs=2, norm_type=(0, 0)):
        # нормировка и изменение размера
        X = np.asarray(X)
        Y = np.asarray(Y)

        if True:
            if norm_type[0] == 1:
                X, minv, maxv = batch_normalize(X)
            else:
                X, minv, maxv = point_normalize(X)

        #  if norm_type[1] == 1:
        #      Y_sumphase, minv, maxv = batch_normalize(Y_sumphase, safe_parametrs=True)
        #   else:
        #    Y_sumphase, minv, maxv = point_normalize(Y_sumphase, safe_parametrs=True)
        # print(X.shape)
        X_ = X.reshape(-1, outputs, self.IMG_SIZE, self.IMG_SIZE)
        Y_sumphase_ = Y_sumphase.reshape(-1, 1, self.IMG_SIZE, self.IMG_SIZE)
        Y_ = Y
        X_train = torch.from_numpy(X_).float()
        y_train = torch.from_numpy(Y_).float()
        Y_sumphase_fin = torch.from_numpy(Y_sumphase_).float()
        return X_train, y_train, Y_sumphase_fin  # , minv, maxv

    def get_traintest_data(self, path, outputs=2, img_deap=1, proportion=(0.9, 0.1, 0)):
        img1, img2, sum_phase, coeffs = [], [], [], []
        # получаем выборку
        with open(path, 'rb') as f:
            img1_, img2_, sum_phase_, _, coeffs = pickle.load(f)
        for i in range(len(img1_)):
            img1.append(cv2.resize(img1_[i], (self.IMG_SIZE, self.IMG_SIZE)))
            img2.append(cv2.resize(img2_[i], (self.IMG_SIZE, self.IMG_SIZE)))
            sum_phase.append(cv2.resize(sum_phase_[i], (self.IMG_SIZE, self.IMG_SIZE)))
        print(len(img2))
        img1, img2, sum_phase, coeffs = np.asarray(img1), np.asarray(img2), np.asarray(sum_phase), np.asarray(coeffs)
        img1 = img1.reshape(-1, 1, self.IMG_SIZE, self.IMG_SIZE)
        img2 = img2.reshape(-1, img_deap, self.IMG_SIZE, self.IMG_SIZE)
        # print(img1.shape, " 4")
        if outputs > 1:
            print(img1.shape)
            print(img2.shape)
            X = np.append(img1, img2, axis=1)
            print(X.shape)
        else:
            X = img1
            print(X.shape)
        y_coeffs = coeffs[:self.IMG_NUM]
        y_sum_phase = sum_phase[:self.IMG_NUM]
        X = X[:self.IMG_NUM]

        train_num, val_num, test_num = (int(self.IMG_NUM * proportion[0]),
                                        int(self.IMG_NUM * proportion[1]),
                                        int(self.IMG_NUM * proportion[2]))

        print(val_num)
        good_indx = filter_good_idx(y_coeffs)
        np.random.shuffle(good_indx)
        print(good_indx)
        X_train, y_train, y_train_sumphase = self.preprocessing(X[good_indx][:train_num],
                                                           y_coeffs[good_indx][:train_num],
                                                           y_sum_phase[good_indx][:train_num],
                                                           outputs)

        X_val, y_val, y_val_sumphase = self.preprocessing(X[good_indx][train_num:train_num + val_num],
                                                     y_coeffs[good_indx][train_num:train_num + val_num],
                                                     y_sum_phase[good_indx][train_num:train_num + val_num],
                                                     outputs)
        if test_num != 0:
            X_test, y_test, y_test_sumphase = self.preprocessing(
                X[good_indx][train_num + val_num:train_num + val_num + test_num],
                y_coeffs[good_indx][train_num + val_num:train_num + val_num + test_num],
                y_sum_phase[good_indx][train_num:train_num + val_num],
                outputs)
            return (X_train, y_train, y_train_sumphase,
                    X_val, y_val, y_val_sumphase,
                    X_test, y_test, y_test_sumphase)

        return (X_train, y_train, y_train_sumphase,
                X_val, y_val, y_val_sumphase)


