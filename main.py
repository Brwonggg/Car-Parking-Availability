import cv2 as cv
from draw_rect import draw_rectangle, set_image
from empty import read_coords, count_empty, coords_exist
from model import Model
from train_step import train_step
from test_step import test_step
from data_sorter import organise_data, ParkingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch, os
from torch import nn

TEST_IMG = '/Users/brandon/Desktop/aerial 2.avif'

sample_folder = '/Users/brandon/Downloads/archive/spots/empty'  
sample_file = os.listdir(sample_folder)[0]
sample_img = cv.imread(os.path.join(sample_folder, sample_file))

img = cv.imread(TEST_IMG)
set_image(img) 

base_path = '/Users/brandon/Downloads/archive/spots'
empty_folder = [os.path.join(base_path, 'empty')]
occupied_folder = [os.path.join(base_path, 'parked')]

train_data = empty_folder + occupied_folder

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
])

if not coords_exist():
    print("No saved coordinates found. Please draw your parking spots.")
    img = cv.imread(TEST_IMG)
    set_image(img)
    cv.namedWindow("Image")
    cv.setMouseCallback("Image", draw_rectangle)
    cv.imshow("Image", img)
    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()
else:
    print("Using previously saved coordinates.")


model = Model()

device = torch.device("cpu")
model = model.to(device)

X_train, X_test, y_train, y_test = organise_data()

class_counts = torch.tensor([
    (y_train == 0).sum().float(),
    (y_train == 1).sum().float()
])
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * 2
loss_fn = nn.CrossEntropyLoss()

train_dataset = ParkingDataset(X_train, y_train, transform=train_transform)
test_dataset = ParkingDataset(X_test, y_test, transform=None)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_step(X_train, y_train, loss_fn)

test_step(X_test,y_test, loss_fn)

coords = read_coords()

count_empty(coords, model, device)


