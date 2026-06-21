import cv2 as cv
from empty import read_coords, count_empty, coords_exist
from model import Model
from test_step import test_step
from data_sorter import organise_data, ParkingDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch, os
from torch import nn
from draw_bounds import draw_row_chunk

TEST_IMG = '/Users/brandon/Downloads/archive/sample_frames/07.png'

sample_folder = '/Users/brandon/Downloads/archive/spots/empty'
sample_file = os.listdir(sample_folder)[0]
sample_img = cv.imread(os.path.join(sample_folder, sample_file))

def set_image(image):
    global img
    img = image

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
    print("No saved coordinates found. Please draw your parking spots, one row/band at a time.")
    
    keep_drawing = True
    while keep_drawing:
        num_spots_per_subrow = int(input("Number of columns of individual parking lots in each row: "))
        num_sub_rows = int(input("Number of rows: "))
        
        draw_row_chunk(TEST_IMG, num_spots_per_subrow, num_sub_rows)
        while True:  # ADDED: keep asking until a valid answer is given
            more = input("Draw another band? (y/n): ").strip().lower()
            if more == "y":
                keep_drawing = True
                break
            elif more == "n":
                keep_drawing = False
                break
            else:
                pass 
            
    print("Finished drawing all bands.")
else:
    print("Using previously saved coordinates.")

model = Model()
device = torch.device("cpu")
model = model.to(device)

X_train, X_test, y_train, y_test = organise_data()

loss_fn = nn.CrossEntropyLoss()

test_dataset = ParkingDataset(X_test, y_test, transform=None)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model.load_state_dict(torch.load('best_model.pth'))
model.eval()

test_step(X_test, y_test, loss_fn)

coords = read_coords()

count_empty(coords, model, device)