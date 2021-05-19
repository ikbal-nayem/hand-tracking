import mediapipe as mp
import cv2


class Hands:
	def __init__(self, max_num_hands=2):
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands)


	def detectHands(self, image, draw=False):
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.landmarks = self.hands.process(rgb)
		if draw and self.landmarks.multi_hand_landmarks:
			for landmark_list in self.landmarks.multi_hand_landmarks:
				self.mp_drawing.draw_landmarks(image, landmark_list, self.mp_hands.HAND_CONNECTIONS)
		return image


	def detectPosition(self, img, handNo=0, draw=True):
		lmList = []
		if self.landmarks.multi_hand_landmarks:
			myHand = self.landmarks.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				lmList.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 1, (255, 0, 255), cv2.FILLED)
		return lmList