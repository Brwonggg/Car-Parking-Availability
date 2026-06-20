import ast
import cv2 as cv
import os

TEST_IMG = '/Users/brandon/Desktop/aerial 2.avif'
base_path = '/Users/brandon/Downloads/archive/spots'
empty_folder = [os.path.join(base_path, 'empty')]
occupied_folder = [os.path.join(base_path, 'parked')]

def read_coords_from_file(filepath):
    with open(filepath, "r") as file:
        content = file.read()
    cleaned = content.replace("Top left:", "").replace("Bottom right:", "")
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    tuples = [ast.literal_eval(line) for line in lines]
    coords = [[tuples[i], tuples[i+1]] for i in range(0, len(tuples), 2)]
    return coords

def show_numbered_spots(image_path, coords, output_path='numbered_spots.png'):
    img = cv.imread(image_path)
    for i, spot in enumerate(coords):
        cv.rectangle(img, spot[0], spot[1], (0, 255, 0), 1)
        cv.putText(img, str(i), spot[0], cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv.imwrite(output_path, img)
    print(f"Saved numbered preview to {output_path}")

aerial_coords = read_coords_from_file('aerial_coords.txt')
show_numbered_spots(TEST_IMG, aerial_coords)

def add_to_existing_dataset(image_path, coords, empty_indices, occupied_indices, empty_folder, occupied_folder):
    image = cv.imread(image_path)
    for i in empty_indices:
        x1, y1 = coords[i][0]
        x2, y2 = coords[i][1]
        crop = image[y1:y2, x1:x2]
        cv.imwrite(os.path.join(empty_folder, f"walmart_{i}.png"), crop)
    for i in occupied_indices:
        x1, y1 = coords[i][0]
        x2, y2 = coords[i][1]
        crop = image[y1:y2, x1:x2]
        cv.imwrite(os.path.join(occupied_folder, f"walmart_{i}.png"), crop)
    print(f"Added {len(empty_indices)} empty + {len(occupied_indices)} occupied images to dataset")

# FILL THESE IN after looking at numbered_spots.png
empty_indices = []      # e.g. [2, 5, 9, 14, 23]
occupied_indices = []   # e.g. [0, 1, 3, 4, 6, 7]

add_to_existing_dataset(
    TEST_IMG,
    aerial_coords,
    empty_indices,
    occupied_indices,
    empty_folder[0],     # your existing folder path
    occupied_folder[0]   # your existing folder path
)