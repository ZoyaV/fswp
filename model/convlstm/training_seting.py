# import os
# from google.colab import drive
# drive.mount('/content/gdrive')

# import sys
# import gc

# py_file_location = '/content/gdrive/My Drive/ao_prediction/notebooks'
# sys.path.append(os.path.abspath(py_file_location))

class Settings():
  def __init__(self):
    self.use_amp = False
    self.n_gpus = 1
    self.lr =  1e-3
    self.beta_1 = 0.9
    self.beta_2 = 0.98
    self.batch_size = 8
    self.epochs = 30
    self.ahead = 20
    self.predict =5
    self.n_hidden_dim = 16
    self.train_path = '/content/gdrive/My Drive/ao_prediction/data/solar_dataset_100x100.pickle'
    self.test_path = '/content/gdrive/My Drive/ao_prediction/data/solar_dataset_100x100.pickle'
    self.val_path = '/content/gdrive/My Drive/ao_prediction/data/solar_dataset_100x100.pickle'

  def asdict(self):
    return {"use_amp":self.use_amp,
            "n_gpu":self.n_gpus,
            "lr":self.lr,
            "beta_1":self.beta_1,
            "beta_2":self.beta_2,
            "batch_size":self.batch_size,
            "epochs":self.epochs,
            "ahead":self.ahead,
            "predict":self.predict,
            "n_hidden_dim":self.n_hidden_dim,
            "train_path":self.test_path,
            "test_path":self.test_path,
            "val_path":self.val_path}

  def fromdict(self, dict):
    self.use_amp = dict["use_amp"]
    self.n_gpu = dict["n_gpus"]
    self.lr = dict["lr"]
    self.beta_1 = dict["beta_1"]
    self.beta_2 = dict["beta_2"]
    self.batch_size = dict["batch_size"]
    self.epochs = dict["epochs"]
    self.ahead = dict["ahead"]
    self.predict = dict["predict"]
    self.n_hidden_dim = dict["n_hidden_dim"]
    self.train_path = dict["test_path"]
    self.test_path = dict["test_path"]
    self.val_path = dict["val_path"]
    return


opt = Settings()