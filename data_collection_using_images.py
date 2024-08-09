
import mediapipe as mp 
import numpy as np 
import cv2 
import os
import time 

def inFrame(lst):
	if lst[28].visibility > 0.4 and lst[27].visibility > 0.4 and lst[15].visibility > 0.4 and lst[16].visibility > 0.4:
		return True 
	return False

# Set the path to the directory containing images
image_dir = "data/sukhasana"

name = input("Enter the name of the Asana: ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

# Loop through each image in the directory
for image_name in os.listdir(image_dir):
	if image_name.endswith(".png") or image_name.endswith(".jpg"):  # Check if the file is an image
		image_path = os.path.join(image_dir, image_name)
		frm = cv2.imread(image_path)

		# Preprocess the image
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
		# print(res.pose_landmarks)
		# print(res.pose_landmarks.landmark)
		if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
			lst = []
			for i in res.pose_landmarks.landmark:
				print(i.x,res.pose_landmarks.landmark[0].x,i.x - res.pose_landmarks.landmark[0].x)
				print(i.y,res.pose_landmarks.landmark[0].y,i.y - res.pose_landmarks.landmark[0].y)
				lst.append(i.x - res.pose_landmarks.landmark[0].x)
				lst.append(i.y - res.pose_landmarks.landmark[0].y)

			X.append(lst)
			print(data_size,"added   ",image_name)
			data_size += 1
			# if data_size==54:
			# 	break
			time.sleep(0.6)
			

		else: 
			cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
			time.sleep(0.6)

		drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)

		cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

		cv2.imshow("window", frm)

		if cv2.waitKey(1) == 27 or data_size > 200:
			break

cv2.destroyAllWindows()

np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)
