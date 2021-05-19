import mediapipe as mp
import cv2


class HandTracker:
	def __init__(self, max_num_hands=2, detection_conf=0.70):
		self.mp_drawing = mp.solutions.drawing_utils
		self.mp_hands = mp.solutions.hands
		self.hands = self.mp_hands.Hands(max_num_hands=max_num_hands, min_detection_confidence=detection_conf)
		self.fingers_top = [4, 8, 12, 16, 20]


	def detectHands(self, image, draw=False):
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		self.landmarks = self.hands.process(rgb)
		if draw and self.landmarks.multi_hand_landmarks:
			for landmark_list in self.landmarks.multi_hand_landmarks:
				self.mp_drawing.draw_landmarks(image, landmark_list, self.mp_hands.HAND_CONNECTIONS)
		return len(self.landmarks.multi_hand_landmarks) if self.landmarks.multi_hand_landmarks else 0


	def detectPosition(self, img, hand_no=0, draw=False):
		landmark_list = []
		if self.landmarks.multi_hand_landmarks:
			hand = self.landmarks.multi_hand_landmarks[hand_no]
			for id, lm in enumerate(hand.landmark):
				h, w, c = img.shape
				cx, cy = int(lm.x * w), int(lm.y * h)
				landmark_list.append([id, cx, cy])
				if draw:
					cv2.circle(img, (cx, cy), 1, (255, 0, 100), cv2.FILLED)
		return landmark_list


	def checkFingersUpOrDown(self, landmark_list):
		finger_list = [0]
		if self.landmarks.multi_hand_landmarks:
			for idx in range(1, len(self.fingers_top)):
				if landmark_list[self.fingers_top[idx]][2] < landmark_list[self.fingers_top[idx]-2][2]:
					finger_list.append(1)
				else: finger_list.append(0)
		return finger_list