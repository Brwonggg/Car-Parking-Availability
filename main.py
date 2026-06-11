import cv2 as cv
from rectangle import draw_rectangle

drawing = False
ix, iy = -1, -1

img = cv.imread('/Users/brandon/Desktop/carparking.jpg')

cv.namedWindow("Image")
cv.setMouseCallback("Image", draw_rectangle)
cv.imshow("Image",img)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()