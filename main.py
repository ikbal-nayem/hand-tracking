import cv2
from hand_detector import HandTracker
import pyautogui


hands = HandTracker(2)
# image = cv2.imread('hand2.jpg')
capture = cv2.VideoCapture(0)
capture.set(3, 640)
capture.set(4, 480)
while True:
	check, image = capture.read()

	num_of_hands = hands.detectHands(image, draw=True)
	if num_of_hands>0:
		landmarks = hands.detectPosition(image, hand_no=0)
		fingers = hands.checkFingersUpOrDown(landmarks)
		print(fingers)
		# if fingers.count(1) == 0:
		# 	print('Back')
		# 	pyautogui.keyDown('down')
		# else:
		# 	print('Front')
		# 	pyautogui.keyDown('up')

	cv2.imshow('Show Image', image)
	if cv2.waitKey(1) & 0xFF == ord('q'):
	  break


capture.release()
cv2.destroyAllWindows()
