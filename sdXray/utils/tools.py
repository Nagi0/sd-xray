import os
import tensorflow as tf
from utils.data_loader import DataLoader, LungSegDataset, Covid19Dataset
from utils.models import UNet, load_model
import matplotlib.pyplot as plt


class Tools:

    @staticmethod
    def train_segmentation(
        data_loader: LungSegDataset | Covid19Dataset,
        img_shape: tuple,
        activation: str,
        kernel_init: str,
        loss: str,
        optimizer: str,
        val_size: float,
        val_metric: str,
        epochs: int,
    ):
        train_dataset, val_dataset = data_loader.build_dataset(val_size)
        print(train_dataset, val_dataset)

        unet = UNet(img_shape, img_shape, activation, kernel_init, loss, optimizer)
        unet.build_model()
        unet.model.summary()

        unet.fit(train_dataset, val_dataset, epochs, val_metric, os.environ["SEGMENTATION_MODEL_FILENAME"])

    @staticmethod
    def test_segmentation(
        model_file_path: str,
        dataset_path: str,
        metadata_path: str,
        input_shape: tuple,
        batch_size: int,
    ):
        data_loader = DataLoader(
            dataset_path,
            metadata_path,
            input_shape,
            batch_size,
        )
        test_dataset = data_loader.build_test_img_dataset()

        loaded_model = load_model(model_file_path, {"dice_coef": UNet.dice_coef})

        if loaded_model is not None:
            loaded_model.summary()
            results = loaded_model.predict(test_dataset, batch_size=batch_size)
            print(tf.shape(results))

            for images in test_dataset.take(1):
                for i, img in enumerate(images):
                    image_np = img.numpy()

                    # Imagem
                    plt.subplot(4, 2, 2 * i + 1)
                    plt.imshow(image_np, cmap="gray")
                    plt.title("Imagem")
                    plt.axis("off")

                    # Máscara
                    plt.subplot(4, 2, 2 * i + 2)
                    plt.imshow(results[i], cmap="gray")
                    plt.title("Máscara")
                    plt.axis("off")

            plt.tight_layout()
            plt.show()
