import numpy as np


class ILSVRCPredictor:
    def __init__(self, class_index) -> None:
        self.class_index = class_index

    def predict_max(self, out):
        maxid = np.argmax(out.detach().numpy())
        predicted_label = self.class_index[str(maxid)][1]
        return predicted_label
