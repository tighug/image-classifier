import json
from PIL import Image
from torch import Tensor
from predictor.ilsvrc_predictor import ILSVRCPredictor
from data.base_transform import BaseTransform
from torchvision import models


def main() -> None:
    CLASS_INDEX_PATH = "./data/imagenet_class_index.json"
    class_index: dict[str, list[str]] = json.load(open(CLASS_INDEX_PATH, "r"))
    IMG_PATH = "./data/goldenretriever-3724972_640.jpg"
    img: Image = Image.open(IMG_PATH)

    RESIZE = 224
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    transform = BaseTransform(RESIZE, MEAN, STD)
    transformed_img: Tensor = transform(img)
    inputs = transformed_img.unsqueeze_(0)

    predictor = ILSVRCPredictor(class_index)
    net = models.vgg16(pretrained=True)
    net.eval()  # 推論モードに設定
    output = net(inputs)
    result = predictor.predict_max(output)

    print(result)


if __name__ == "__main__":
    main()
