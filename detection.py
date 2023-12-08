import cv2
from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(
    "yolov8n.pt")  # load a pretrained model (recommended for training)

source = cv2.imread("./bus.jpg")
# Use the model
results = model(source)  # predict one image
for result in results:
    for xyxy in (result.boxes.xyxy):
        rect = xyxy.numpy()
        cv2.rectangle(source,
                      (int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])),
                      (0, 255, 0), 1)
    print(result.probs)
cv2.imshow("Image", source)

cv2.waitKey(0)
cv2.destroyAllWindows()
