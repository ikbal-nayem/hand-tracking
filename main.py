import cv2
from hand_detector import HandTracker


hands = HandTracker(2)
image = cv2.imread('hand2.jpg')
num_of_hands = hands.detectHands(image, draw=True)
if num_of_hands>0:
	landmarks = hands.detectPosition(image, hand_no=0)
	fingers = hands.checkFingersUpOrDown(landmarks)
	print(fingers)
	cv2.imshow('Show Image', image)
else:
	print('No hands found!')


cv2.waitKey(0)
cv2.destroyAllWindows()
