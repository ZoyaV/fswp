
import pickle
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from training_seting import opt


def calc_split_settings(data, ep_len = 25):
    total_len = len(data[0])
    image_shape = data[0][0].shape
    axes = len(data)
    blocks = total_len//ep_len
  #  print("total_len", total_len)
   # print("total blocks", blocks)
    total_batches = blocks // delimetr(blocks,minv=10, maxv=15)[0]
  #  print("total_batches", total_batches)
    one_batch = total_batches // delimetr(total_batches,minv=4, maxv=12)[-1]
  #  print("one_batch", one_batch)
  #  print(blocks//one_batch, one_batch, ep_len)
    total = blocks//one_batch* one_batch*ep_len
  #  print("Totoal len", blocks//one_batch * one_batch * ep_len)
    if (blocks//one_batch * one_batch * ep_len)!=total_len:
   #     print("Coudls split Data! - part of data!")
        pass
    return total_len - total, axes, blocks//one_batch, one_batch, ep_len, *image_shape, 1

def reshape_data(data):
    mlenght, axes, batches, blen, ep_len, w, h, imax = calc_split_settings(data, ep_len = 25)
    new_data = []
  #  print("mlen", mlenght)
    axes = axes-1 if axes==4 else axes
    for ax in range(axes):
      #  print(ax)
      #  print(data[ax][mlenght:].shape)
        try:
            new_data.append(data[ax][mlenght:].reshape(batches, blen, ep_len, w, h, imax))
        except:
            print(data[ax][mlenght:][0])
    return new_data

def delimetr(numb, minv, maxv):
    d = []

    for i in range(numb - 1, 1, -1):
        if (numb % i == 0):
            if i < maxv and i>minv:
                d.append(i)
    return d

class WaveFrontLoader():
    def __init__(self, data_path, ep_len, trainp = 0.8):
        self.data_path = data_path
        self.ep_len = ep_len
        self.axes = 3
        self.trainp = trainp
        self.init()
        
    def init(self):
        with open(self.data_path, 'rb') as f:
            self.data = pickle.load(f)[:10000]

    def __next__(self):
        data = reshape_data(self.data)
        self.axes = len(data)
        nums = len(data[0])
        returns = []
        for ax in range(self.axes):
         #   data[ax] = data[ax][1:] - data[ax][:-1]
            np.random.shuffle(data[ax])
        
        nums = int(nums * self.trainp)
       # print("BNUN - ", nums)
        
        for n in range(nums):
            for ax in range(self.axes):
               # data_ax = np.random.shuffle(data[ax], axis = 0)
                np.random.shuffle(data[ax][n])
                data_ax_n = data[ax][n]
                returns.append(data_ax_n)
            yield returns

class WaveFrontLoaderNorm(WaveFrontLoader):
    def __next__(self):
        gen =  super().__next__()
        for data in gen:
            for ax in range(self.axes):
               # print(data.shape)
              #  data[ax] = data[ax].numpy()
                mins = data[ax].min(axis=(1,2,3,4))
                maxs = data[ax].max(axis=(1,2,3,4))
                data[ax] = (data[ax] - mins.reshape(7,1,1,1,1))/(maxs.reshape(7,1,1,1,1) - mins.reshape(7,1,1,1,1))
              #  data[ax] = torch.from_numpy(data[ax]).float()
            yield data