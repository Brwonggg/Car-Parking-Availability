import cv2 as cv

drawing = False
ix, iy = -1, -1
img = None

def set_image(image):
    global img
    img = image

def draw_rectangle(event, x, y, flags, params):
    global ix, iy, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_rect = img.copy()
            cv.rectangle(img_rect, (ix,iy), (x,y), (0,255,0), 2)
            cv.imshow("Image",img_rect)
    
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(img, (ix, iy), (x,y), (0,255,0), 2)
        cv.imshow("Image",img)

        with open("coords.txt","a") as file:
            file.write(f"Top left:{(ix,iy)}\n")
            file.write(f"Bottom right:{(x,y)}\n")