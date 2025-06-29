import os
from dotenv import load_dotenv
from utils.tools import Tools
from utils.data_loader import CoCross, LungSegDataset
from utils.models import UNet


if __name__ == "__main__":
    load_dotenv("./config/.env")
    mode = os.environ["MODE"]
    # data_loader = CoCross("CoCross", "CoCross/CoCrossDatabase.xlsx")
    # survival_data = data_loader.get_survival_data()
    # print(survival_data)

    if mode == "train_segmentation":
        Tools.train_segmentation(
            "sdXray/datasets/lung-segmentation-dataset",
            "sdXray/datasets/lung-segmentation-dataset/train.csv",
            (256, 256, 1),
            "ReLU",
            "he_normal",
            "binary_focal_crossentropy",
            "adam",
            0.25,
            "val_binary_io_u",
            20,
            4,
        )

    elif mode == "validate_segmentation":
        pass
