import os
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


@dataclass
class DataLoader:
    path: str
    metadata: str
    data_shape: tuple
    batch_size: int

    def get_metadata(self):
        if "csv" in self.metadata:
            metadata_file = pd.read_csv(self.metadata)
        else:
            metadata_file = pd.read_excel(self.metadata)

        return metadata_file

    def get_img_data(self):
        images_list = []
        metadata = self.get_metadata()
        for element in metadata.iterrows():
            img = element[1]["images"]

            if os.path.exists(f"{self.path}/{img}"):
                images_list.append(f"{self.path}/{img}")

        return pd.DataFrame({"image": images_list})

    def load_image(self, image_path):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=self.data_shape[2])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (self.data_shape[0], self.data_shape[1]))

        return image

    def build_test_img_dataset(self):
        images_path_df = self.get_img_data()
        image_paths = images_path_df["image"].to_list()

        test_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        test_dataset = test_dataset.map(self.load_image, num_parallel_calls=tf.data.AUTOTUNE)
        test_dataset = test_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        return test_dataset


@dataclass
class CoCross(DataLoader):

    def get_imgs_paths(self, keep: str):
        base_dir = Path(self.path)
        base_dir = sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda x: int(x.name))
        image_map = {}

        for id_dir in base_dir:
            xr_dir = id_dir / "XR"
            if xr_dir.exists() and xr_dir.is_dir():
                image_files = sorted([f for f in xr_dir.iterdir() if f.suffix in [".jpg", ".png", ".jpeg"]])
                if image_files:
                    if keep == "last":
                        image_map[int(id_dir.name)] = str(image_files[-1])
                    elif keep == "first":
                        image_map[int(id_dir.name)] = str(image_files[0])
                    else:
                        image_map[int(id_dir.name)] = str(image_files)

        return image_map

    def get_survival_data(self, keep: str = "last"):
        metadata_df = self.get_metadata()
        metadata_df = metadata_df.groupby("ID").last()
        image_map = self.get_imgs_paths(keep)
        metadata_df["images"] = metadata_df.index.map(image_map)
        metadata_df = metadata_df[["images", "hospitalization days"]]

        return metadata_df


@dataclass
class LungSegDataset(DataLoader):

    def get_segmentation_data(self):
        images_list = []
        masks_list = []
        metadata = self.get_metadata()
        for element in metadata.iterrows():
            img = element[1]["images"]
            mask = element[1]["masks"]

            if os.path.exists(f"{self.path}/{img}") and os.path.exists(f"{self.path}/{mask}"):
                images_list.append(f"{self.path}/{img}")
                masks_list.append(f"{self.path}/{mask}")

        return pd.DataFrame({"image": images_list, "mask": masks_list})

    @staticmethod
    def show_samples(dataset, num_samples=3):
        plt.figure(figsize=(10, num_samples * 3))

        for i, (image, mask) in enumerate(dataset.take(num_samples)):
            image_np = tf.squeeze(image).numpy()
            mask_np = tf.squeeze(mask).numpy()

            # Imagem
            plt.subplot(num_samples, 2, 2 * i + 1)
            plt.imshow(image_np, cmap="gray")
            plt.title("Imagem")
            plt.axis("off")

            # Máscara
            plt.subplot(num_samples, 2, 2 * i + 2)
            plt.imshow(mask_np, cmap="gray")
            plt.title("Máscara")
            plt.axis("off")

        plt.tight_layout()
        plt.show()

    def load_image_and_mask(self, image_path, mask_path):
        # Lê e decodifica a imagem original
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, channels=self.data_shape[2])
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, (self.data_shape[0], self.data_shape[1]))

        # Lê e decodifica a máscara
        mask = tf.io.read_file(mask_path)
        mask = tf.image.decode_png(mask, channels=self.data_shape[2])
        mask = tf.cast(mask > 0, tf.float32)
        mask = tf.image.resize(mask, (self.data_shape[0], self.data_shape[1]), method="nearest")

        return image, mask

    def build_dataset(self, test_size: float, show_sample: bool = False):
        images_path_df = self.get_segmentation_data()
        image_paths = images_path_df["image"].to_list()
        mask_paths = images_path_df["mask"].to_list()

        train_image_path, val_image_path, train_mask_path, val_mask_path = train_test_split(
            image_paths, mask_paths, test_size=test_size
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((train_image_path, train_mask_path))
        train_dataset = train_dataset.map(self.load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((val_image_path, val_mask_path))
        val_dataset = val_dataset.map(self.load_image_and_mask, num_parallel_calls=tf.data.AUTOTUNE)

        train_dataset = train_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

        if show_sample:
            self.show_samples(train_dataset)

        return train_dataset, val_dataset
