import cv2 as cv
from empty import read_coords, count_empty, coords_exist
from test_step import test_step
from train_step import train_step
import torch, os
from draw_bounds import draw_row_chunk
from model import Model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, 'data', 'spots')
empty_folder = [os.path.join(base_path, 'empty')]
occupied_folder = [os.path.join(base_path, 'parked')]
train_data = empty_folder + occupied_folder

TEST_IMG = os.path.join(BASE_DIR, 'data', 'test_images', 'test_parking.png')
COORDS_FILE = os.path.join(BASE_DIR, 'coords.txt')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pth')

# device = torch.device("cpu")

# model, X_test, y_test, test_loader, loss_fn, xent_metric = train_step(train_data, device, MODEL_PATH)
# model = test_step(model, test_loader, loss_fn, xent_metric, device, MODEL_PATH)

if not os.path.exists(MODEL_PATH):
    print(f"No trained model found at {MODEL_PATH}. Training now...")
    device = torch.device("cpu")
    model, X_test, y_test, test_loader, loss_fn, xent_metric = train_step(train_data, device, MODEL_PATH)
    model = test_step(model, test_loader, loss_fn, xent_metric, device, MODEL_PATH)
else:
    print(f"Found existing trained model, loading it directly.")
    device = torch.device("cpu")
    model = Model().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

img = cv.imread(TEST_IMG)

if not coords_exist(COORDS_FILE):
    print("No saved coordinates found. Please draw your parking spots, one row/band at a time.")
    
    keep_drawing = True
    while keep_drawing:
        num_spots_per_subrow = int(input("Number of columns of individual parking lots in each row: "))
        num_sub_rows = int(input("Number of rows: "))
        
        draw_row_chunk(TEST_IMG, num_spots_per_subrow, num_sub_rows, COORDS_FILE)
        while True:
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

coords = read_coords(COORDS_FILE)

count_empty(coords, model, device, TEST_IMG)