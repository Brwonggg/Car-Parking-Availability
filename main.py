import cv2 as cv
from draw_rect import draw_rectangle, set_image

img = cv.imread('/Users/brandon/Desktop/carparking.jpg')
set_image(img) 

cv.namedWindow("Image")
cv.setMouseCallback("Image", draw_rectangle)
cv.imshow("Image",img)

while True:
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cv.destroyAllWindows()