import cv2 as cv
from draw_rect import draw_rectangle, set_image
from empty import read_coords, count_empty
from model import Model
from train_step import train_step
from test_step import test_step
from data_sorter import organise_data, ParkingDataset
from torch.utils.data import DataLoader
from torchvision import transforms

img = cv.imread('/Users/brandon/Desktop/carparking.jpg')
set_image(img) 

train_data = ['/Users/brandon/Downloads/parking/clf-data/empty',
              '/Users/brandon/Downloads/parking/clf-data/not_empty'
              ]

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

cv.namedWindow("Image")
cv.setMouseCallback("Image", draw_rectangle)
cv.imshow("Image",img)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()

model = Model(train_data)

X_train, X_test, y_train, y_test = organise_data()
train_dataset = ParkingDataset(X_train, y_train, transform=train_transform)
test_dataset = ParkingDataset(X_test, y_test, transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

train_step(X_train, y_train)
test_step(X_test,y_test)

coords = read_coords()

count_empty(coords)

