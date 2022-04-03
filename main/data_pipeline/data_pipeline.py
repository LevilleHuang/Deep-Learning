from abc import ABC, abstractmethod


class DataPipeline(ABC):

    @property
    @abstractmethod
    def train_data(self):
        pass

    @property
    @abstractmethod
    def val_data(self):
        pass

    @property
    @abstractmethod
    def test_data(self):
        pass