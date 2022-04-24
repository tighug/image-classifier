from PIL import Image
from torch import Tensor
from torchvision import transforms


class BaseTransform:
    def __init__(
        self,
        resize: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.base_transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

    def __call__(self, img: Image) -> Tensor:
        return self.base_transform(img)
