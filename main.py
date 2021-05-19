import cv2
from process import Hands


hands = Hands(4)
image = cv2.imread('hand1.jpg')
image = hands.detectHands(image)
landmarks = hands.detectPosition(image, draw=True)
cv2.imshow('Show Image', image)


cv2.waitKey(0)
cv2.destroyAllWindows()
