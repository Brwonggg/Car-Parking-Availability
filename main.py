import cv2 as cv
from draw_rect import draw_rectangle, set_image
from empty import read_coords, count_empty
from model import Model
from train_step import train_step
from test_step import test_step
from data_sorter import organise_data, ParkingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import time

start_time = time.time()

#'/Users/brandon/Desktop/empty-parking-lots-aerial-view-600nw-1841895190.webp'
#'/Users/brandon/Desktop/carparking.jpg'
#'/Users/brandon/Desktop/parkingarea.png'
img = cv.imread('/Users/brandon/Desktop/parkingarea.png')
#img = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
set_image(img) 

train_data = ['/Users/brandon/Downloads/matchbox_cars_parkinglot/empty',
              '/Users/brandon/Downloads/matchbox_cars_parkinglot/occupied']

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

open("coords.txt", "w").close()

cv.namedWindow("Image")
cv.setMouseCallback("Image", draw_rectangle)
cv.imshow("Image",img)

end_time = time.time()

print(end_time - start_time)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()

model = Model()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

X_train, X_test, y_train, y_test = organise_data()
train_dataset = ParkingDataset(X_train, y_train, transform=train_transform)
test_dataset = ParkingDataset(X_test, y_test, transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_step(X_train, y_train)
test_step(X_test,y_test)

coords = read_coords()

count_empty(coords)


