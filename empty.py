import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader

model = Model()

coords = [fill this out with coords]

def detect_empty(image, spot):
    x1, y1 = spot[0]
    x2, y2 = spot[1]
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0]:
        print("Invalid coordinates")
        return False
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        print("Empty ROI")
        return False
    gray_roi = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    resized_roi = cv.resize(gray_roi, (48,48))
    resized_roi = resized_roi.astype('float32') / 255
    resized_roi = np.expand_dims(resized_roi, axis=0)
    resized_roi = np.expand_dims(resized_roi, axis=-1)
    
    pred = model...
    threshold = 0.01
    if pred[0][0] > threshold:
        return True
    else:
        return False 
    
current_image = cv.imread('/Users/brandon/Desktop/carparking.jpg')
empty_spots = 0

for spot in coords:
    if detect_empty(current_image, spot):
        cv.rectangle(current_image, spot[0], spot[1], (0,255,0), 2)
        empty_spots += 1
    else:
        cv.rectangle(current_image, spot[0], spot[1], (255,0,0), 2)

font = cv.FONT_HERSHEY_SIMPLEX
cv.putText(current_image, f"Empty spots: {empty_spots}", (50,50), font, 1.5, (255,255,255), 3, cv.LINE_AA)

cv.imshow("Parking Lot", current_image)
cv.waitkey(0)
cv.destroyAllWindows() 
