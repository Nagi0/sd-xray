import os
from dotenv import load_dotenv
from utils.tools import Tools
from utils.data_loader import LungSegDataset, CoCross, Covid19Dataset


if __name__ == "__main__":
    load_dotenv("./config/.env")
    mode = os.environ["MODE"]

    if mode == "train_segmentation":
        data_shape = (256, 256, 1)

        data_loader = Covid19Dataset(
            "sdXray/datasets/COVID-19_Radiography_Dataset",
            "sdXray/datasets/COVID-19_Radiography_Dataset/COVID19DatabaseImage.csv",
            data_shape,
            4,
        )
        Tools.train_segmentation(
            data_loader,
            data_shape,
            "ReLU",
            "he_normal",
            "binary_focal_crossentropy",
            "adam",
            0.25,
            "val_binary_io_u",
            30,
        )

    elif mode == "test_segmentation":
        Tools.test_segmentation(
            "./model_lung_segmentation.h5", ".", "sdXray/datasets/CoCross/CoCrossDatabaseImage.csv", (256, 256, 1), 4
        )

    elif mode == "generate_survival_img_metadata":
        data_loader = CoCross(
            "sdXray/datasets/CoCross", "sdXray/datasets/CoCross/CoCrossDatabase.xlsx", (256, 256, 1), 1
        )
        survival_data = data_loader.get_survival_data()
        survival_data.to_csv("sdXray/datasets/CoCross/CoCrossDatabaseImage.csv", index=False)

    elif mode == "generate_covid_img_metadata":
        data_loader = Covid19Dataset(
            "sdXray/datasets/COVID-19_Radiography_Dataset",
            "",
            (256, 256, 1),
            1,
        )
        image_data = data_loader.get_segmentation_data()
        image_data.to_csv("sdXray/datasets/COVID-19_Radiography_Dataset/COVID19DatabaseImage.csv", index=False)
