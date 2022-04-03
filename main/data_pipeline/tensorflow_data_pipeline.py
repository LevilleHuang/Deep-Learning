import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from . import data_pipeline


class ImageClassificationDataPipeline(data_pipeline.DataPipeline):

    def __init__(self, params):
        self.path  = params.pop("data_root")
        self.train = pd.read_csv(
            params.pop("train_csv"),
            names=["filename", "class"]
        ).astype(str)
        self.val   = pd.read_csv(
            params.pop("val_csv"),
            names=["filename", "class"]
        ).astype(str)
        self.test  = pd.read_csv(
            params.pop("test_csv"),
            names=["filename", "class"]
        ).astype(str)
        self.extra = params.pop("extra")

        self.train_gen = ImageDataGenerator(
            **params["data_augmentation"],
            **params["preprocessing"]
        )
        self.eval_gen  = ImageDataGenerator(**params["preprocessing"])

    @property
    def train_data(self):
        return self.train_gen.flow_from_dataframe(
            dataframe=self.train,
            directory=self.path,
            shuffle=True,
            **self.extra
        )

    @property
    def val_data(self):
        return self.eval_gen.flow_from_dataframe(
            dataframe=self.val,
            directory=self.path,
            shuffle=False,
            **self.extra
        )

    @property
    def test_data(self):
        return self.eval_gen.flow_from_dataframe(
            dataframe=self.test,
            directory=self.path,
            shuffle=False,
            **self.extra
        )
