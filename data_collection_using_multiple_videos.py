import mediapipe as mp 
import numpy as np 
import cv2 
import os

def inFrame(lst):
    if lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and lst[15].visibility>0.6 and lst[16].visibility>0.6:
        return True 
    return False

# Directory containing the video files
video_dir = "data/bhajungasan/"
# List of video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

name = input("Enter the name of the Asana: ")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

for video_file in video_files:
    print(video_file,"video_file")
    cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
    while True:
        lst = []
        _, frm = cap.read()

        if frm is None:
            break
        
        # frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            X.append(lst)
            data_size += 1
            print(f"{data_size} frames added from {video_file}")
        else: 
            cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)
        cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27 or data_size > 150:  # Option to limit the number of frames
            break

    cap.release()

cv2.destroyAllWindows()
np.save(f"{name}.npy", np.array(X))
print(f"Data saved. Shape: {np.array(X).shape}")
