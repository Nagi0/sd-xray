import os
from sdXray.utils.data_loader import DataLoader
from utils.models import UNet, load_model
import matplotlib.pyplot as plt


def train_segmentation(
    data_loader: DataLoader,
    img_shape: tuple,
    activation: str,
    kernel_init: str,
    loss: str,
    optimizer: str,
    val_size: float,
    val_metric: str,
    epochs: int,
):
    train_dataset, val_dataset = data_loader.build_train_segmentation(val_size)

    model = UNet(img_shape, img_shape, activation, kernel_init, loss, optimizer)
    model.build_model()

    model.fit(train_dataset, val_dataset, epochs, val_metric, os.environ["SEGMENTATION_MODEL_FILENAME"])


def val_segmentation(
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
    dataset = data_loader.build_val_segmentation()
    loaded_model = load_model(model_file_path, {"dice_coef": UNet._dice_coef})
    if loaded_model is not None:
        loaded_model.evaluate(dataset, batch_size=batch_size)
        results = loaded_model.predict(dataset, batch_size=batch_size)

        for images in dataset.take(1):
            for i, img in enumerate(images[0]):
                image_np = img.numpy()

                _plot_image_and_mask(image_np, results[i], i)

        plt.tight_layout()
        plt.show()


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
    dataset = data_loader.build_test_segmentation()

    loaded_model = load_model(model_file_path, {"dice_coef": UNet._dice_coef})

    if loaded_model is not None:
        results = loaded_model.predict(dataset, batch_size=batch_size)

        for images in dataset.take(1):
            for i, img in enumerate(images):
                image_np = img.numpy()
                _plot_image_and_mask(image_np, results[i], i)

        plt.tight_layout()
        plt.show()


def _plot_image_and_mask(image, mask, index):
    plt.subplot(4, 2, 2 * index + 1)
    plt.imshow(image, cmap="gray")
    plt.title("Imagem")
    plt.axis("off")

    plt.subplot(4, 2, 2 * index + 2)
    plt.imshow(mask, cmap="gray")
    plt.title("Máscara")
    plt.axis("off")
