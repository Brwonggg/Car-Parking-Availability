import cv2 as cv
import numpy as np
from model import Model
import ast
import torch
from torchvision import transforms

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied']

model = Model()

def read_coords():
    with open("coords.txt", "r") as file:
        content = file.read()
    cleaned = content.replace("Top left:","").replace("Bottom right:","")

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    tuples = [ast.literal_eval(line) for line in lines]

    coords = [[tuples[i], tuples[i+1]] for i in range(0, len(tuples), 2)]

    return coords

def predict_spot(roi_tensor):
    model.eval()
    with torch.inference_mode():
        pred = model(roi_tensor)
        return pred

def detect_if_empty(image, coords):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Invalid coordinates")
        return None
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("Empty ROI")
        return None
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    resized_roi = cv.resize(gray_roi, (48, 48))
    resized_roi = resized_roi.astype('float32') / 255
    resized_roi = np.expand_dims(resized_roi, axis=0)   # add channel dim: (1,48,48)
    resized_roi = np.expand_dims(resized_roi, axis=0)   # add batch dim: (1,1,48,48)
    roi_tensor = torch.FloatTensor(resized_roi)
    return roi_tensor

def predict_empty(roi_tensor, threshold=0.01):
    pred = predict_spot(roi_tensor)
    probabilities = torch.softmax(pred, dim=1)
    empty_probability = probabilities[0][0].item()  # probability of class 0 (empty)
    return empty_probability > threshold

def count_empty(coords):   
    #'/Users/brandon/Desktop/carparking.jpg'
    #'/Users/brandon/Desktop/empty-parking-lots-aerial-view-600nw-1841895190.webp'
    #'/Users/brandon/Desktop/parkingarea.png'
    current_image = cv.imread('/Users/brandon/Desktop/parkingarea.png')
    #current_image = cv.rotate(current_image, cv.ROTATE_90_CLOCKWISE)
    empty_spots = 0

    for spot in coords:
        roi_tensor = detect_if_empty(current_image, spot)
        if roi_tensor is not None and predict_empty(roi_tensor, threshold= 0.01):
            cv.rectangle(current_image, spot[0], spot[1], (0,255,0), 2)
            empty_spots += 1
        else:
            cv.rectangle(current_image, spot[0], spot[1], (0,0,255), 2)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(current_image, f"Empty spots: {empty_spots}", (50,50), font, 1.5, (255,255,255), 3, cv.LINE_AA)

    cv.imshow("Parking Lot", current_image)
    cv.waitKey(500)
    cv.waitKey(0)
    cv.destroyAllWindows() 


