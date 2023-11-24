import torch
from PIL import Image
import cv2
import numpy as np


def draw_bounding_box(img, class_id, confidence, xmin, ymin, xmax, ymax):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (xmin,ymin), (xmax,ymax), color, 2)
    cv2.putText(img, f'{label} ({confidence:.2f})', (xmin-10,ymin-10),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


# Load pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:v6.0', 'yolov5s', pretrained=True)

# Load the test image
img_path = 'C:\\Users\\jesus\\OneDrive\\Documentos\\Master\\pic\\ejemplo.jpg'

classes = model.names

# generate different colors for different classes 
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

# Process the image
# tamaño =  640 
# img_o = Image.open(img_path)  # Load image using PIL
# img = img_o.resize((tamaño, tamaño))  # Resize image to half of its original size
# # img = img_o.resize((width // 2, height // 2))  # Resize image to half of its original size
# width, height = img_o.size
# escala_x = width / tamaño
# escala_y = height / tamaño

img = Image.open(img_path)  # Load image using PIL

results = model(img)  # Pass the image to the model
print(results.pandas().xyxy[0])  # Print results as pandas dataframe
detections = results.xyxy[0].numpy()  # Get detections as numpy array


# Convert PIL Image to OpenCV format
img_cv = np.array(img) 
img_cv = img_cv[:, :, ::-1].copy()

labels = []
scores = []
bboxes = []
for result in detections:
    # print(result)
    labels.append(int(result[5]))
    scores.append(result[4])
    bboxes.append(result[:4].astype(int))  # Convert bbox coordinates to integers


conf_threshold = 0.1
nms_threshold = 0.8
# apply non-max suppression
indices = cv2.dnn.NMSBoxes(bboxes, scores, conf_threshold, nms_threshold)

# go through the detections remaining
# after nms and draw bounding box
for i in indices:
    box = bboxes[i]
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]
    
    draw_bounding_box(img_cv, labels[i], scores[i], round(xmin), round(ymin), round(xmax), round(ymax))


cv2.imshow('YOLO Detection', img_cv)
cv2.waitKey(0)
cv2.destroyAllWindows()
