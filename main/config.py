import os

class PathConfig:

    def __init__(self, data_root):
        self.image = os.path.join(data_root, "image")
        self.label = os.path.join(data_root, "label.csv")
        self.train = os.path.join(data_root, "train.csv")
        self.val   = os.path.join(data_root, "val.csv")
        self.test  = os.path.join(data_root, "test.csv")