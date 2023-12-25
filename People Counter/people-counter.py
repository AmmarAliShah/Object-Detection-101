import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap = cv2.VideoCapture("../Videos/people.mp4")
model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("mask.png")
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limitUp = [103, 161, 296, 161]
limitDown = [527, 489, 735, 489]
totalCountUp = []
totalCountDown = []

while True:
    success, img = cap.read()
    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    cvzone.overlayPNG(img, imgGraphics, pos=[730, 260])
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Object Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Confidence
            conf = box.conf[0]
            conf = math.ceil(conf*100)/100
            # cvzone.putTextRect(img, f'{conf}', (max(0,x1), max(35, y1)))

            # Classification
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass == "person" and conf > 0.3:
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    trackerResults = tracker.update(detections)
    cv2.line(img, (limitUp[0], limitUp[1]), (limitUp[2], limitUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitDown[0], limitDown[1]), (limitDown[2], limitDown[3]), (0, 0, 255), 5)

    for result in trackerResults:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        print(result)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
        cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)), scale=2, thickness=1)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitUp[0] < cx < limitUp[2] and limitUp[1]-15 < cy < limitUp[3]+15:
            if totalCountUp.count(Id) == 0:
                totalCountUp.append(Id)
                cv2.line(img, (limitUp[0], limitUp[1]), (limitUp[2], limitUp[3]), (0, 255, 0), 5)

        if limitDown[0] < cx < limitDown[2] and limitDown[1]-15 < cy < limitDown[3]+15:
            if totalCountDown.count(Id) == 0:
                totalCountDown.append(Id)
                cv2.line(img, (limitDown[0], limitDown[1]), (limitDown[2], limitDown[3]), (0, 255, 0), 5)

    # cvzone.putTextRect(img, f'Count: {len(totalCount)}', (50, 50))
    cv2.putText(img, str(len(totalCountUp)), (929, 345), cv2.FONT_HERSHEY_PLAIN, 5, (139, 195, 75), 7)
    cv2.putText(img, str(len(totalCountDown)), (1191, 345), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 230), 7)

    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion)
    cv2.waitKey(1)
