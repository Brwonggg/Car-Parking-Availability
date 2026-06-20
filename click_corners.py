import cv2 as cv

clicked_points = []
TEST_IMG = '/Users/brandon/Desktop/aerial 2.avif'

img = cv.imread(TEST_IMG)

def click_corners(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv.circle(img, (x, y), 4, (0, 255, 0), -1)
        cv.imshow("Calibrate", img)

cv.namedWindow("Calibrate")
cv.setMouseCallback("Calibrate", click_corners)
cv.imshow("Calibrate", img)
print("Click top-left, top-right, bottom-left, bottom-right of each row, in order. Press 'q' when done.")

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()

rows = [clicked_points[i:i+4] for i in range(0, len(clicked_points), 4)]

def interpolate_row_spots(top_left, top_right, bottom_left, bottom_right, num_spots):
    spots = []
    for i in range(num_spots):
        frac_start = i / num_spots
        frac_end = (i + 1) / num_spots
        x1 = int(top_left[0] + (top_right[0] - top_left[0]) * frac_start)
        x2 = int(top_left[0] + (top_right[0] - top_left[0]) * frac_end)
        y1 = top_left[1]
        y2 = bottom_left[1]
        spots.append([(x1, y1), (x2, y2)])
    return spots

all_spots = []
num_spots_per_row = 20  # adjust based on actual spot count per row
for row in rows:
    top_left, top_right, bottom_left, bottom_right = row
    all_spots.extend(interpolate_row_spots(top_left, top_right, bottom_left, bottom_right, num_spots_per_row))

# Save in the SAME text format as your existing coords.txt, so read_coords() needs no changes at all
with open("aerial_coords.txt", "w") as f:
    for spot in all_spots:
        top_left, bottom_right = spot
        f.write(f"Top left:{top_left}\n")
        f.write(f"Bottom right:{bottom_right}\n")

print(f"Saved {len(all_spots)} spot coordinates to aerial_coords.txt")