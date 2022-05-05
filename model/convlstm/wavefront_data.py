import pickle
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
# from google.colab import drive
# drive.mount('/content/gdrive')

# import os
# import sys
# py_file_location = '/content/gdrive/My Drive/ao_prediction/notebooks'
# sys.path.append(os.path.abspath(py_file_location))

from training_seting import opt


def preprocessing(X):
   
    X = np.asarray(X)
    X_train = torch.from_numpy(X).float()
    return X_train


class WavefrontData():
    SLICE = 1
    IMG_SIZE = 128

    def __init__(self, path, shuffle=False, type_f = "SCREEN", batch_size=20, data_last=15, data_ahead=5, axis=0, test=False,
                 alternative=None, screen_imgs = 1):
        self.path = path
        self.type_f = type_f
        self.index = 0
        self.X_train = []
        self.y_train = []
        self.y_train_dat = []
        self.batch_size = opt.batch_size
        self.data_last = opt.ahead
        self.data_ahead = opt.predict
        self.axis = axis
        self.screen = []
        self.test = False
        self.data = []
        self.screen_imgs = screen_imgs
        self.alternative_data = None
        if alternative is not None:
            self.load_from_source(alternative)
        else:
            self.__load()
    def dropna(self,X,screen_imgs=1):
      #  print(X.shape,screen_imgs )
        X_new = []
        idx = []
        if screen_imgs == 2:
          for i,x in enumerate(X):
            for j,x_ in enumerate(x):
              X[i][j] = cv2.resize(x_[self.SLICE:-self.SLICE, self.SLICE:-self.SLICE], (self.IMG_SIZE,self.IMG_SIZE))              
              if not np.sum(np.isnan(X[i][j])):
              #X_new.append(img)
                idx.append(i)
          X = np.asarray(X)
        #  print(X.shape, " _0")
          return X[idx],idx
        else:
          for i,x in enumerate(X):
             X[i] = cv2.resize(x[self.SLICE:-self.SLICE, self.SLICE:-self.SLICE], (self.IMG_SIZE,self.IMG_SIZE))
             if not np.sum(np.isnan(X[i])):
              #X_new.append(img)
                idx.append(i)
          X = np.asarray(X)
        #  print(X.shape, " !0")
          return X[idx],idx

    def __load(self):
        with open(self.path, 'rb') as f:
            screen, img_2, sum_phase, Y = pickle.load(f)
            screen, img_2, sum_phase, Y =  screen[:5000], img_2[:5000], sum_phase[:5000], Y[:5000]
            if self.screen_imgs == 2:
              screen = np.append(screen[:,np.newaxis], img_2[:,np.newaxis], axis = 1)
              print("Screen shape", screen.shape)
        if self.type_f == "SCREEN":
              X_train = screen     
        if self.type_f == "PHASE":
            X_train = sum_phase
        if self.type_f == "SCREENPHAS":
            X_train = sum_phase
            dropnaX,idx = self.dropna(X_train)
            self.screen = screen[idx]
          #  print(self.screen.shape , " __0")
            dropnaS,idx = self.dropna(self.screen, screen_imgs=self.screen_imgs)
           # print(dropnaS.shape, "_0_")
            dropnaX = dropnaX
            self.X_train = preprocessing(dropnaX)
            self.data = self.X_train
            self.screen = preprocessing(dropnaS)
            return

        dropnaX,_ = self.dropna(X_train)
      #  print("Xtain ", X_train.shape)
       # print("Dropnax ", np.asarray(dropnaX).shape)
        self.X_train = preprocessing(dropnaX)
        self.data = self.X_train
      #  print("Data shape" , self.data.shape)
      #  print("X train shape", self.X_train.shape)

    def normalizate_pics(self, data):
        indx = []
        # self.y_train = (self.y_train - torch.min(self.y_train))/(torch.max(self.y_train) - torch.min(self.y_train))
        for i in range(len(data)):
       #     data[i] = (data[i] - torch.min(data[i])) / (
          #              torch.max(data[i]) - torch.min(data[i]))
            #print(torch.min(self.y_train[i,0]))
            if ((torch.mean(data[i, 0]) != torch.max(data[i, 0]))
                    and torch.sum(torch.isnan(data[i, 0]).int()) == 0):
              indx.append(i)
        data = data[indx]
        #### Костыль для деления данных на 25
        c = (len(data)//1000) * 1000
        return data[:c], indx[:c]

    def load_from_source(self, source):
        self.alternative_data = source

    def dataset(self):
        return self.data

    def reshaped(self, data, indexes_2 = None, imges_count = 1):
        # print(data.shape)
        if self.axis == 0:
            data, mask = self.normalizate_pics(data)
            #print(data.shape)
            size = self.data_last + self.data_ahead
            splits = np.array_split(data.detach().numpy(), size, axis=0)
            #print(np.asarray(splits).shape)
            #print(size)
            #  print(self.data)
            data = np.asarray(splits).reshape(-1, size, self.IMG_SIZE, self.IMG_SIZE, imges_count)
            if not indexes_2:
                indexes = list(range(data.shape[0]))
                np.random.shuffle(indexes)
            else:
                indexes = indexes_2
            data = data[indexes]
            # for i in range(data.shape[0]):
            #   data[i] = (data[i] - np.min(data[i]))/(np.max(data[i]) - np.min(data[i]))
            data = np.asarray(np.array_split(data, self.batch_size, axis=0)).reshape(-1, self.batch_size,
                                                                                               size, self.IMG_SIZE, self.IMG_SIZE, imges_count)
            data = torch.from_numpy(data)
        return data, indexes, mask

    def __iter__(self):
        if len(self.screen)!=0:              
               # print("data shape: ", self.data.shape)
              #  print("screen shape: ", self.screen.shape)
               # plt.imshow(self.data[0])
               # plt.show()
                data,indexes, mask = self.reshaped(self.data,imges_count=1)
               # plt.imshow(self.screen[0][0])
               # plt.show()
                screen_1, _, mask2 = self.reshaped(self.screen[mask,0],imges_count=1)
                screen_2, _, mask3 = self.reshaped(self.screen[mask,1],imges_count=1)
              #  print("fin data shape: ", data.shape)
              #  print("fin screen shape: ", screen_1.shape)
              #  print("fin screen shape: ", screen_2.shape)

               # plt.imshow(screen_1[0,0,0,:,:,0])
               # plt.show()
                data = data.detach().numpy()
                screen_1 = screen_1.detach().numpy()
                screen_2 = screen_2.detach().numpy()                
                
                data = np.append(data, screen_1, axis = 5)
                data = np.append(data, screen_2, axis = 5)
                data = preprocessing(data)

              #  self.data[:,1],_ = self.reshaped(data[:,0],indexes)
               # np.random.shuffle(data)
                return iter(data)
        if self.alternative_data is not None:
            data = self.alternative_data
        else:
            data,_,mask_ = self.reshaped(self.data)
            

            
        np.random.shuffle(data)
       
        return iter(data)
        
