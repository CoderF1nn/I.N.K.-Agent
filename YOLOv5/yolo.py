import cv2
import numpy as np
import torch


class ImageTemplate:
    def __init__(self, image):
        self.image = image

    def get_template(self):
        return self.image

class YoloPrediction:
    def __init__(self, prediction: torch.Tensor):
        self.prediction = prediction
        pass

    def get_prediction(self):
        return self.prediction

    def get_annotated_image(self):
        img = self.prediction.render()[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def show(self, wait: int = 1, window_name: str = "YOLO"):
        img = self.get_annotated_image()
        cv2.imshow(window_name, img)
        cv2.waitKey(wait)

class YoloDetection:
    def __init__(self, model_path: str = "YOLOv5/YoloModels/cuphead_goopy_model.pt"):
        self.model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=model_path, source="github"
        )
        pass

    def process_objects(self):
        TempVector = np.full((4, 4), float(-1))
        for object in self.detectedobjects:
            xmin, ymin, xmax, ymax, confidence, objectindex = object
            objectindex = int(objectindex)
            if float(TempVector[objectindex][-2]) < confidence:
                TempVector[objectindex] = [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    confidence,
                    objectindex,
                ]
        ConfFiltVector = np.full((2, 4), float(-1))
        # set first item to cuphead
        ConfFiltVector[0] = TempVector[0]
        for object in TempVector:
            xval, yval, confidence, objectindex = object
            if (confidence > ConfFiltVector[1][2]) and objectindex != 0:
                if ConfFiltVector[0][0] != -1:
                    xval = ConfFiltVector[0][0] - xval
                else:
                    xval = 0.0
                if ConfFiltVector[0][1] != -1:
                    yval = ConfFiltVector[0][1] - yval
                else:
                    yval = 0.0
                ConfFiltVector[1] = [xval, yval, confidence, objectindex]
        ConfFiltVector = np.delete(ConfFiltVector, [2, 3], axis=1)
        return ConfFiltVector

    def detect_single_image(self, image: ImageTemplate | cv2.Mat) -> YoloPrediction:
        labelindex = ["Cuphead", "GoopyPhase1", "GoopyPhase2", "GoopyPhase3"]
        if isinstance(image, ImageTemplate):
            image_mat = image.get_template()
        else:
            image_mat = image

        prediction = self.model(image_mat, size=640)
        self.detectedobjects = prediction.xyxyn[0][:, :].numpy()

        return YoloPrediction(prediction)
