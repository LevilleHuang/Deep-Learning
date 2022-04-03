from tensorflow.keras import applications, optimizers
from . import learning_pipeline


class ImageClassificationLearningPipeline(learning_pipeline.LearningPipeline):
    available_optimizers = {
        "sgd": optimizers.SGD,
        "rmsprop": optimizers.RMSprop,
        "adam": optimizers.Adam,
        "adagrad": optimizers.Adagrad,
        "adamax": optimizers.Adamax,
        "Adadelta": optimizers.Adadelta,
        "nadam": optimizers.Nadam
    }
    available_models = {
        "densenet121": applications.densenet.DenseNet121,
        "densenet169": applications.densenet.DenseNet169,
        "densenet201": applications.densenet.DenseNet201,
        "efficientnetb0": applications.efficientnet.EfficientNetB0,
        "efficientnetb1": applications.efficientnet.EfficientNetB1,
        "efficientnetb2": applications.efficientnet.EfficientNetB2,
        "efficientnetb3": applications.efficientnet.EfficientNetB3,
        "efficientnetb4": applications.efficientnet.EfficientNetB4,
        "efficientnetb5": applications.efficientnet.EfficientNetB5,
        "efficientnetb6": applications.efficientnet.EfficientNetB6,
        "efficientnetv2b7": applications.efficientnet.EfficientNetB7,
        # "efficientnetv2b0": applications.efficientnet_v2.EfficientNetV2B0,
        # "efficientnetv2b1": applications.efficientnet_v2.EfficientNetV2B1,
        # "efficientnetv2b2": applications.efficientnet_v2.EfficientNetV2B2,
        # "efficientnetv2b3": applications.efficientnet_v2.EfficientNetV2B3,
        # "efficientnetv2l": applications.efficientnet_v2.EfficientNetV2L,
        # "efficientnetv2m": applications.efficientnet_v2.EfficientNetV2M,
        # "efficientnetv2s": applications.efficientnet_v2.EfficientNetV2S,
        "inceptionresnetv2": applications.inception_resnet_v2.InceptionResNetV2,
        "inceptionv3": applications.inception_v3.InceptionV3,
        "mobilenet": applications.mobilenet.MobileNet,
        "mobilenetv2": applications.mobilenet_v2.MobileNetV2,
        # "mobilenetv3": applications.mobilenet_v3.MobileNetV3,
        "nasnetlarge": applications.nasnet.NASNetLarge,
        "nasnetmobile": applications.nasnet.NASNetMobile,
        "resnet50": applications.resnet.ResNet50,
        "resnet101": applications.resnet.ResNet101,
        "resnet152": applications.resnet.ResNet152,
        "resnet50v2": applications.resnet_v2.ResNet50V2,
        "resnet101v2": applications.resnet_v2.ResNet101V2,
        "resnet152v2": applications.resnet_v2.ResNet152V2,
        "vgg16": applications.vgg16.VGG16,
        "vgg19": applications.vgg19.VGG19,
        "xception": applications.xception.Xception
    }

    def __init__(self, hparams):
        self.model     = self.available_models[hparams["model"]["name"]]
        self.model     = self.model(**hparams["model"]["params"])
        self.optimizer = self.available_optimizers[hparams["optimizer"]["name"]]
        self.optimizer = self.optimizer(**hparams["optimizer"]["params"])
        self.loss      = hparams["loss"]
        self.metrics   = hparams["metrics"]
        self.callbacks = []

        if hparams["model"]["checkpoint"] is not None:
            self.model.load_weights(hparams["model"]["checkpoint"])

        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=self.metrics
        )

    def train(self, epochs, train_data, val_data=None, initial_epoch=0):
        self.model.fit(
            train_data,
            epochs=epochs,
            validation_data=val_data,
            callbacks=self.callbacks,
            initial_epoch=initial_epoch
        )

    def test(self, test_data):
        return self.model.evaluate(test_data)

    def predict(self, data):
        return self.model.predict(data)

    def attach(self, callback):
        self.callbacks.append(callback)

    def detach(self, callback):
        self.callbacks.remove(callback)
