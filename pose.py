import cv2
import numpy as np
from ultralytics import YOLO

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
            [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
            [2, 4], [3, 5], [4, 6], [5, 7]]
pose_palette = np.array(
    [[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
     [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
     [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
     [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
     [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
    dtype=np.uint8)
kpt_color = pose_palette[[
    16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9
]]
limb_color = pose_palette[[
    9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16
]]

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO(
    "yolov8n-pose.pt")  # load a pretrained model (recommended for training)

source = cv2.imread("./bus.jpg")
# Use the model
results = model(source)  # predict one image
for result in results:
    names = result.names
    boxes = result.boxes.data.tolist()
    # keypoints.data.shape -> n,17,3
    keypoints = result.keypoints.cpu().numpy()

    # keypoint -> 每个人的关键点
    for keypoint in keypoints.data:
        for i, (x, y, conf) in enumerate(keypoint):
            #color_k = [int(x) for x in kpt_color[i]]
            color_k = (0, 255, 0)
            if conf < 0.5:
                continue
            if x != 0 and y != 0:
                cv2.circle(source, (int(x), int(y)),
                           5,
                           color_k,
                           -1,
                           lineType=cv2.LINE_AA)
        for i, sk in enumerate(skeleton):
            pos1 = (int(keypoint[(sk[0] - 1), 0]), int(keypoint[(sk[0] - 1),
                                                                1]))
            pos2 = (int(keypoint[(sk[1] - 1), 0]), int(keypoint[(sk[1] - 1),
                                                                1]))

            conf1 = keypoint[(sk[0] - 1), 2]
            conf2 = keypoint[(sk[1] - 1), 2]
            if conf1 < 0.5 or conf2 < 0.5:
                continue
            if pos1[0] == 0 or pos1[1] == 0 or pos2[0] == 0 or pos2[1] == 0:
                continue
            cv2.line(source,
                     pos1,
                     pos2, [int(x) for x in limb_color[i]],
                     thickness=2,
                     lineType=cv2.LINE_AA)

    # for box in boxes:
    #     left, top, right, bottom = int(box[0]), int(box[1]), int(box[2]), int(
    #         box[3])
    #     confidence = box[4]
    #     label = int(box[5])

    #     cv2.rectangle(source, (left, top), (right, bottom),
    #                   color=(0, 255, 0),
    #                   thickness=2,
    #                   lineType=cv2.LINE_AA)
    #     caption = f"{names[label]} {confidence:.2f}"
    #     w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
    #     cv2.rectangle(source, (left - 3, top - 33), (left + w + 10, top),
    #                   (0, 255, 0), -1)
    #     cv2.putText(source, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)

cv2.imshow("Image", source)

cv2.waitKey(0)
cv2.destroyAllWindows()
