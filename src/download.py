import os
from zipfile import ZipFile
import requests

DATA_DIR = "./data/"
IMG_DATA_URL = "https://download.pytorch.org/tutorial/hymenoptera_data.zip"
IMG_DATA_ZIP_NAME = "hymenoptera_data.zip"
CLASS_INDEX_URL = "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
CLASS_INDEX_NAME = "imagenet_class_index.json"


def _download(url: str, dist_path: str) -> None:
    res: requests.Response = requests.get(url)

    with open(dist_path, "wb") as dir:
        dir.write(res.content)


def _extract_zip(zip_path: str, dist_path: str) -> None:
    with ZipFile(zip_path, "r") as f:
        f.extractall(path=dist_path)


def main():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    IMG_DIST_PATH = os.path.join(DATA_DIR, IMG_DATA_ZIP_NAME)
    INDEX_DIST_PATH = os.path.join(DATA_DIR, CLASS_INDEX_NAME)

    _download(IMG_DATA_URL, IMG_DIST_PATH)
    _extract_zip(IMG_DIST_PATH, DATA_DIR)
    _download(CLASS_INDEX_URL, INDEX_DIST_PATH)
    os.remove(IMG_DIST_PATH)


if __name__ == "__main__":
    main()
