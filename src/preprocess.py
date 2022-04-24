from PIL import Image
from matplotlib import pyplot
import numpy as np
from data.base_transform import BaseTransform

IMG_PATH = "./data/goldenretriever-3724972_640.jpg"


def main():
    img = Image.open(IMG_PATH)
    pyplot.imshow(img)
    pyplot.show()

    resize = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = BaseTransform(resize, mean, std)
    img_transformed = transform(img)  # torch.Size([3, 224, 224])

    # (色、高さ、幅)を (高さ、幅、色)に変換し、0-1に値を制限して表示
    img_transformed = img_transformed.numpy().transpose((1, 2, 0))
    img_transformed = np.clip(img_transformed, 0, 1)
    pyplot.imshow(img_transformed)
    pyplot.show()


if __name__ == "__main__":
    main()
