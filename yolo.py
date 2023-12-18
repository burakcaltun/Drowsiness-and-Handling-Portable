import time
import cv2.dnn
import numpy as np

class YOLO:
    def __init__(self, config, model, labels, size=416, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold
        self.size = size
        self.outputLayerNames = []
        self.labels = labels

        try:
            self.net = cv2.dnn.readNetFromDarknet(config, model)
        except:
            raise ValueError("Error initializing YOLO network")

        layerNames = self.net.getLayerNames()
        for i in self.net.getUnconnectedOutLayers():
            self.outputLayerNames.append(layerNames[int(i) - 1])

    def inferenceFromFile(self, file):
        image = cv2.imread(file)
        return self.inference(image)

    def inference(self, display):
        Dheight, Dweight = display.shape[0:2]
        blobFormating = cv2.dnn.blobFromImage(display, 1 / 255.0, (self.size, self.size), swapRB=True, crop=False)

        self.net.setInput(blobFormating)
        start = time.time()
        layerOutputs = self.net.forward(self.outputLayerNames)
        end = time.time()
        timeScale = end - start

        boxes = []
        confidences = []
        classIDs = []

        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confidence:
                    box = detection[0:4] * np.array([Dweight, Dheight, Dweight, Dheight])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x, y = (boxes[i][0], boxes[i][1])
                w, h = (boxes[i][2], boxes[i][3])
                id = classIDs[i]
                confidence = confidences[i]

                results.append((id, self.labels[id], confidence, x, y, w, h))

        return Dweight, Dheight, timeScale, results



