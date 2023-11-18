import cv2
import numpy as np
from xml.etree import ElementTree as ET

# Load the pre-trained YOLOv3 neural network
net = cv2.dnn.readNet('yolov3/yolov3.weights', 'yolov3/yolov3.cfg')

# Get the names of the output layers of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the input image
img = cv2.imread('../Dataset/test/apple_78.jpg')

# Resize the image to a fixed size for processing
img = cv2.resize(img, None, fx=0.4, fy=0.4)

# Get the image dimensions
height, width, channels = img.shape

# Prepare the input image for the neural network
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# Set the input for the neural network
net.setInput(blob)

# Run the forward pass to get the network's output
outs = net.forward(output_layers)

# Threshold for confidence
confidence_threshold = 0.5

# Apply Non-Maximum Suppression (NMS)
def non_max_suppression(boxes, overlap_threshold=0.3):
    if len(boxes) == 0:
        return []

    # Convert [x, y, w, h] bounding boxes to [x1, y1, x2, y2]
    boxes = np.array(boxes)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]

    # Compute the area of bounding boxes and sort the indices by the bottom-right y-coordinate of the boxes
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = np.argsort(y2)

    selected_boxes = []
    while len(indices) > 0:
        last = len(indices) - 1
        i = indices[last]
        selected_boxes.append(i)

        xx1 = np.maximum(x1[i], x1[indices[:last]])
        yy1 = np.maximum(y1[i], y1[indices[:last]])
        xx2 = np.minimum(x2[i], x2[indices[:last]])
        yy2 = np.minimum(y2[i], y2[indices[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        intersection = w * h
        iou = intersection / (area[i] + area[indices[:last]] - intersection)

        indices = np.delete(indices, np.concatenate(([last], np.where(iou > overlap_threshold)[0])))

    return boxes[selected_boxes].astype("int")

# Loop over each of the detected objects
boxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > confidence_threshold:
            # Get the coordinates of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2

            # Add the bounding box coordinates and confidence to the lists
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Perform Non-Maximum Suppression (NMS) to get the final bounding boxes
indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)

# Ensure indices is not empty and proceed with drawing boxes
if len(indices) > 0:
    for i in indices.flatten():
        box = boxes[i]
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'Object {class_ids[i]}', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting image
    cv2.imshow('Detected Objects', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No objects found or all objects suppressed by NMS.")