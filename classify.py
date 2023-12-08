import cv2
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(
    "yolov8n-cls.pt")  # load a pretrained model (recommended for training)

source = cv2.imread("./bus.jpg")
# Use the model
results = model(source)  # predict one image
for result in results:
    print(result.names[result.probs.top1])

cv2.imshow("Image", source)

cv2.waitKey(0)
cv2.destroyAllWindows()
