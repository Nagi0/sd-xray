import os
from dataclasses import dataclass
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMAGE_COLUMN = "image"
IMAGES_FOLDER = "images"
MASK_COLUMN = "mask"
MASKS_FOLDER = "masks"
IMAGE_EXTENSION = ".png"


def show_samples(dataset: tf.data.Dataset, num_samples: int = 4):
    plt.figure(figsize=(10, num_samples * num_samples))

    for image, mask in dataset.take(1):
        images_np = tf.squeeze(image).numpy()
        masks_np = tf.squeeze(mask).numpy()

        for i, (img, mk) in enumerate(zip(images_np, masks_np)):
            # Imagem
            plt.subplot(4, 2, 2 * i + 1)
            plt.imshow(img, cmap="gray")
            plt.title(IMAGE_COLUMN)
            plt.axis("off")

            # Máscara
            plt.subplot(4, 2, 2 * i + 2)
            plt.imshow(mk, cmap="gray")
            plt.title(MASK_COLUMN)
            plt.axis("off")

    plt.tight_layout()
    plt.show()


@dataclass
class DataLoader:
    path: str
    metadata: str
    data_shape: tuple
    batch_size: int

    def build_train_segmentation(self, test_size: float, show_sample: bool = False):
        images_path_df = self._get_segmentation_data()
        image_paths = images_path_df[IMAGE_COLUMN].to_list()
        mask_paths = images_path_df[MASK_COLUMN].to_list()

        train_image_path, val_image_path, train_mask_path, val_mask_path = train_test_split(
            image_paths, mask_paths, test_size=test_size
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_path, train_mask_path))
        train_dataset = train_dataset.map(self._load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_path, val_mask_path))
        val_dataset = val_dataset.map(self._load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        if show_sample:
            show_samples(train_dataset)

        return train_dataset, val_dataset

    def build_val_segmentation(self, show_sample: bool = False):
        images_path_df = self._get_segmentation_data()
        image_paths = images_path_df[IMAGE_COLUMN].to_list()
        mask_paths = images_path_df[MASK_COLUMN].to_list()

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
        dataset = dataset.map(self._load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        if show_sample:
            show_samples(dataset)

        return dataset

    def build_test_segmentation(self):
        images_path_df = self._get_test_img_data()
        image_paths = images_path_df[IMAGE_COLUMN].to_list()

        test_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        test_dataset = test_dataset.map(self._load_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return test_dataset

    def get_img_dataframe(self):
        images = []
        masks = []
        for dirpath, _, filenames in os.walk(self.path):
            dirpath = dirpath.replace("\\", "/")
            for file in filenames:
                if IMAGES_FOLDER in dirpath and file.endswith(IMAGE_EXTENSION):
                    images.append(f"{dirpath}/{file}")
                elif MASKS_FOLDER in dirpath and file.endswith(IMAGE_EXTENSION):
                    masks.append(f"{dirpath}/{file}")

        return pd.DataFrame({IMAGE_COLUMN: images, MASK_COLUMN: masks})

    def _get_segmentation_data(self):
        images_list = []
        masks_list = []
        metadata = self._get_metadata()
        for element in metadata.iterrows():
            img = element[1][IMAGE_COLUMN]
            mask = element[1][MASK_COLUMN]

            if os.path.exists(f"{self.path}/{img}") and os.path.exists(f"{self.path}/{mask}"):
                images_list.append(f"{self.path}/{img}")
                masks_list.append(f"{self.path}/{mask}")

        return pd.DataFrame({IMAGE_COLUMN: images_list, MASK_COLUMN: masks_list})

    def _get_test_img_data(self):
        images_list = []
        metadata = self._get_metadata()
        for element in metadata.iterrows():
            img = element[1][IMAGE_COLUMN]

            if os.path.exists(f"{self.path}/{img}"):
                images_list.append(f"{self.path}/{img}")

        return pd.DataFrame({IMAGE_COLUMN: images_list})

    def _get_metadata(self):
        if "csv" in self.metadata:
            metadata_file = pd.read_csv(self.metadata)
        else:
            metadata_file = pd.read_excel(self.metadata)

        return metadata_file

    def _load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=self.data_shape[2])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (self.data_shape[0], self.data_shape[1]))

        return image

    def _load_image_and_mask(self, image_path, mask_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=self.data_shape[2])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (self.data_shape[0], self.data_shape[1]))

        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=self.data_shape[2])
        mask = tf.cast(mask > 0, tf.float32)
        mask = tf.image.resize(mask, (self.data_shape[0], self.data_shape[1]), method="nearest")

        return image, mask
