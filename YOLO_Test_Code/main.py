import cv2
import numpy as np

# Load the pre-trained YOLOv3 neural network
net = cv2.dnn.readNetFromDarknet('yolov3/yolov3.cfg', 'yolov3/yolov3.weights')

# Get the names of the output layers of the network
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load the input image
img = cv2.imread('test/apple_78.jpg')

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

# Loop over each of the detected objects
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Get the coordinates of the bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = center_x - w // 2
            y = center_y - h // 2

            # Draw the bounding box on the image
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Display the resulting image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()