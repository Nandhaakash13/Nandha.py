import cv2
import numpy as np
import os

# Thresholds and video capture
thres = 0.45  
nms_threshold = 0.5
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 150)

# Load class names
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load model configuration and weights
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Set up the model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success, image = cap.read()
    if not success:
        print("Failed to capture image from webcam")
        break

    # Detect objects in the frame
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    if len(classIds) > 0:
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        # Perform Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        # Ensure indices is not empty before processing
        if len(indices) > 0:
            for i in indices.flatten():  # Use flatten() to avoid scalar indexing error
                box = bbox[i]
                x, y, w, h = box

                # Access the correct class name
                classId = int(classIds[i])
                label = classNames[classId - 1]

                # Draw bounding box and label on the image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(image, label, 
                            (x + 10, y + 30), cv2.FONT_HERSHEY_COMPLEX, 
                            1, (0, 255, 0), 2)

    # Display the output frame
    cv2.imshow("Output", image)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
