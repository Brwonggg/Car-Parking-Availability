import cv2 as cv
import numpy as np
from model import Model
import ast
import torch
from torchvision import transforms
import os


TEST_IMG = '/Users/brandon/Desktop/aerial 2.avif'

base_path = '/Users/brandon/Downloads/archive/spots'
empty_folder = [os.path.join(base_path, 'empty')]
occupied_folder = [os.path.join(base_path, 'parked')]

train_data = empty_folder + occupied_folder

model = Model()

def read_coords():
    with open("coords.txt", "r") as file:
        content = file.read()
    cleaned = content.replace("Top left:","").replace("Bottom right:","")

    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    tuples = [ast.literal_eval(line) for line in lines]

    coords = [[tuples[i], tuples[i+1]] for i in range(0, len(tuples), 2)]

    return coords

def predict_spot(roi_tensor, model):
    model.eval()
    with torch.inference_mode():
        pred = model(roi_tensor)
        return pred

def detect_if_empty(image, coords, device):
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Invalid coordinates")
        return None
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("Empty ROI")
        return None
    resized_roi = cv.resize(roi, (48, 48))
    resized_roi = resized_roi.astype('float32') / 255
    resized_roi = np.transpose(resized_roi, (2, 0, 1))  
    roi_tensor = torch.FloatTensor(resized_roi)
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    roi_tensor = normalize(roi_tensor).unsqueeze(0)
    roi_tensor = roi_tensor.to(device)
    return roi_tensor

def predict_empty(roi_tensor, model, threshold=0.5):
    pred = predict_spot(roi_tensor, model)
    probabilities = torch.softmax(pred, dim=1)
    empty_probability = probabilities[0][0].item()  
    return empty_probability > threshold

def count_empty(coords, model, device):   
    current_image = cv.imread(TEST_IMG)
    empty_spots = 0

    for spot in coords:
        roi_tensor = detect_if_empty(current_image, spot, device)
        if roi_tensor is not None and predict_empty(roi_tensor, model, threshold=0.5):
            cv.rectangle(current_image, spot[0], spot[1], (0,255,0), 2)
            empty_spots += 1
        else:
            cv.rectangle(current_image, spot[0], spot[1], (0,0,255), 2)

    font = cv.FONT_HERSHEY_SIMPLEX
    cv.putText(current_image, f"Empty spots: {empty_spots}", (50,50), font, 1.5, (255,255,255), 3, cv.LINE_AA)

    cv.imshow("Parking Lot", current_image)
    while True:  
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

def coords_exist():
    return os.path.exists("coords.txt") and os.path.getsize("coords.txt") > 0




