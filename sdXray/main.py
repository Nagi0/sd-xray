import os
import logging
from dotenv import load_dotenv
from sdXray.utils.exemples import *
from sdXray.utils.data_loader import DataLoader

DATA_SHAPE = (256, 256, 1)
VAL_SIZE = 0.25
BATCH_SIZE = 4
ACTIVATION = "ReLU"
KERNEL_INIT = "he_normal"
LOSS_FUNCTION = "binary_focal_crossentropy"
OPTIMIZER = "adam"
EPOCHS = 100
VAL_METRIC = "val_binary_io_u"

COVID_DATASET = "sdXray/datasets/COVID-19_Radiography_Dataset"
COVID_DATASET_METADATA = "sdXray/datasets/COVID-19_Radiography_Dataset/COVID19DatabaseImage.csv"
TUBERCULOSIS_DATASET_METADATA = "sdXray/datasets/Tuberculosis Segmentation/images_path.csv"
LUNG_VESSEL_DATASET = "sdXray/datasets/lung_vessel"
LUNG_VESSEL_DATASET_METADATA = "sdXray/datasets/lung_vessel/lung_vessel.csv"


def train_model():
    data_loader = DataLoader(
        COVID_DATASET,
        COVID_DATASET_METADATA,
        DATA_SHAPE,
        BATCH_SIZE,
    )
    train_segmentation(
        data_loader,
        DATA_SHAPE,
        ACTIVATION,
        KERNEL_INIT,
        LOSS_FUNCTION,
        OPTIMIZER,
        VAL_SIZE,
        VAL_METRIC,
        30,
    )


def test_model():
    test_segmentation(
        os.environ["SEGMENTATION_MODEL_FILENAME"],
        ".",
        TUBERCULOSIS_DATASET_METADATA,
        DATA_SHAPE,
        BATCH_SIZE,
    )


def validate_model():
    val_segmentation(
        os.environ["SEGMENTATION_MODEL_FILENAME"],
        ".",
        LUNG_VESSEL_DATASET_METADATA,
        DATA_SHAPE,
        BATCH_SIZE,
    )


def generate_paths_metadata():
    data_loader = DataLoader(
        LUNG_VESSEL_DATASET,
        ".",
        DATA_SHAPE,
        BATCH_SIZE,
    )
    metadata_df = data_loader.get_img_dataframe()
    metadata_df.to_csv(f"{data_loader.path}.csv")


if __name__ == "__main__":
    load_dotenv("./config/.env")
    mode = os.environ["MODE"]
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info(
        "This code is an exemple of usage of the SD-XRAY framework. Make sure to download the required datasets. You can also insert your own dataset, in that case, make sure to create a metadata csv file with the paths of the images and masks"
    )
    mode = input(
        "Chose on the options below:\n[1] Train Segmentation Model\n[2] Test Segmentation Model\n[3] Validate Segmentation Model (labels required)\n[4] Build dataset paths metadata file\nChoice: "
    ).strip()
    mode = int(mode)
    options_dict = {1: train_model, 2: test_model, 3: validate_model, 4: generate_paths_metadata}
    options_dict[mode]()
