import cv2 as cv
import operation

def draw_rectangle(event, x, y, flags, params):
    global ix, iy, drawing
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_rect = operation.img.copy()
            cv.rectangle(img_rect, (ix,iy), (x,y), (0,255,0), 2)
            cv.imshow("Image",img_rect)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        cv.rectangle(operation.img, (ix, iy), (x,y), (0,255,0), 2)
        cv.imshow("Image",operation.img)
        print(f"Top left:{(ix,iy)}" )
        print(f"Bottom right:{(x,y)}")