import cv2 as cv


def draw_row_chunk(image_path, num_spots_per_subrow, num_sub_rows, coords_file):
    """
    Click 4 points bounding the FULL band
    The function then divides horizontally into num_spots_per_subrow columns
    And vertically into num_sub_rows, producing a full grid in one call
    """
    clicked = []
    img = cv.imread(image_path)

    def on_click(event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            clicked.append((x, y))
            cv.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv.imshow("Click 4 corners: TL, TR, BL, BR", img)

    cv.namedWindow("Click 4 corners: TL, TR, BL, BR")
    cv.setMouseCallback("Click 4 corners: TL, TR, BL, BR", on_click)
    cv.imshow("Click 4 corners: TL, TR, BL, BR", img)
    print("Click: top-left, top-right, bottom-left, bottom-right of the FULL band. Press 'q' when done.")

    while True:
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cv.destroyAllWindows()

    top_left, top_right, bottom_left, bottom_right = clicked[0], clicked[1], clicked[2], clicked[3]

    full_top = top_left[1]
    full_bottom = bottom_left[1]
    full_left = top_left[0]
    full_right = top_right[0]

    spots = []
    for row_i in range(num_sub_rows):  
        row_frac_start = row_i / num_sub_rows
        row_frac_end = (row_i + 1) / num_sub_rows
        y1 = int(full_top + (full_bottom - full_top) * row_frac_start)
        y2 = int(full_top + (full_bottom - full_top) * row_frac_end)

        for col_i in range(num_spots_per_subrow):  
            col_frac_start = col_i / num_spots_per_subrow
            col_frac_end = (col_i + 1) / num_spots_per_subrow
            x1 = int(full_left + (full_right - full_left) * col_frac_start)
            x2 = int(full_left + (full_right - full_left) * col_frac_end)
            spots.append([(x1, y1), (x2, y2)])

    with open(coords_file, "a") as f:
        for spot in spots:
            f.write(f"Top left:{spot[0]}\n")
            f.write(f"Bottom right:{spot[1]}\n")

    print(f"Added {len(spots)} spots ({num_sub_rows} sub-rows x {num_spots_per_subrow} columns) to coords.txt")
