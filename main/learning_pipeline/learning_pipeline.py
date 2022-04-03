from abc import ABC, abstractmethod


class LearningPipeline(ABC):

    @abstractmethod
    def train(self, train_data, val_data=None):
        pass

    @abstractmethod
    def test(self, test_data):
        pass

    @abstractmethod
    def predict(self, data):
        pass