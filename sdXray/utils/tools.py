import os
from utils.data_loader import CoCross, LungSegDataset
from utils.models import UNet


class Tools:

    @staticmethod
    def train_segmentation(
        dataset_path: str,
        metadata_path: str,
        img_shape: tuple,
        activation: str,
        kernel_init: str,
        loss: str,
        optimizer: str,
        val_size: float,
        val_metric: str,
        epochs: int,
        batch_size: int,
    ):
        data_loader = LungSegDataset(
            dataset_path,
            metadata_path,
            img_shape,
            batch_size,
        )
        train_dataset, val_dataset = data_loader.build_dataset(val_size)
        print(train_dataset, val_dataset)

        unet = UNet(img_shape, img_shape, activation, kernel_init, loss, optimizer)
        unet.build_model()
        unet.model.summary()

        unet.fit(train_dataset, val_dataset, epochs, val_metric, os.environ["SEGMENTATION_MODEL_FILENAME"])

    @staticmethod
    def validate_segmentation():
        pass
