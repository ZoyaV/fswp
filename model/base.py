import os
import glob
import yaml
from yaml.loader import SafeLoader
import names
import uuid

class BaseTrainer():
    def __init__(self, args_set):
        self.model_base = "BASE"
        self.save_path = "."
        self.run_name = names.get_full_name()
        self.config_path = args_set
        self.args=dict()

    def init_model(self):
        pass

    def train_model(self):
        pass

    def load_configs(self):
        with open(self.config_path) as f:
            data = yaml.load(f, SafeLoader)
        self.args = data
        return self.args

    def load_dataset(self):
        pass

    def buity_name(self):
        filename = str(uuid.uuid4())
        filename = f"{self.model_base}_run_{''.join(self.run_name.split())}_{filename}.pth"
        directory = os.path.join( self.save_path, filename)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        return directory

if __name__ == "__main__":
    baseclass = BaseTrainer("../fswp_train.yaml")