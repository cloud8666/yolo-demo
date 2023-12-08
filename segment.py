import cv2
import numpy as np
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(
    "yolov8n-seg.pt")  # load a pretrained model (recommended for training)

source = cv2.imread("./bus.jpg")
# Use the model
results = model(source)  # predict one image
for result in results:
    for xy in result.masks.xy:
        points = np.array(xy, np.int32)
        cv2.drawContours(source, [points], -1, (0, 255, 0), 2)

cv2.imshow("Image", source)

cv2.waitKey(0)
cv2.destroyAllWindows()
