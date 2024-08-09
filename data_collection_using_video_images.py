import mediapipe as mp 
import numpy as np 
import cv2 
import os

def inFrame(lst):
    if lst[28].visibility > 0.5 and lst[27].visibility > 0.5 and lst[15].visibility>0.50 and lst[16].visibility>0.5:
        return True 
    return False

def inFrame1(lst):
    if lst[28].visibility > 0.35 and lst[27].visibility > 0.35 and lst[15].visibility>0.35 and lst[16].visibility>0.35:
        return True 
    return False

# Directories containing video and image files
video_dir = "data/vriksasana/video"
image_dir = "data/vriksasana"

# List of video and image files in the directories
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# name = input("Enter the name of the Asana: ")
name="VRIKS ASAN"
holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

# Process video files
for video_file in video_files:
    print(video_file)
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
            # print(f"{data_size} frames added from {video_file}")
        else: 
            cv2.putText(frm, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        drawing.draw_landmarks(frm, res.pose_landmarks, holistic.POSE_CONNECTIONS)
        cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("window", frm)

        if cv2.waitKey(1) == 27 or data_size > 150:  # Option to limit the number of frames
            break

    cap.release()

# Process image files
for image_file in image_files:
    # print(image_file)
    lst = []
    img = cv2.imread(os.path.join(image_dir, image_file))
    img = cv2.flip(img, 1)
    res = holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # print("insde image")
    if res.pose_landmarks and inFrame1(res.pose_landmarks.landmark):
        for i in res.pose_landmarks.landmark:
            lst.append(i.x - res.pose_landmarks.landmark[0].x)
            lst.append(i.y - res.pose_landmarks.landmark[0].y)

        X.append(lst)
        data_size += 1
        # print(f"{data_size} frames added from {image_file}")
    else: 
        cv2.putText(img, "Make Sure Full body visible", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    drawing.draw_landmarks(img, res.pose_landmarks, holistic.POSE_CONNECTIONS)
    cv2.putText(img, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("window", img)

    if cv2.waitKey(1) == 27 or data_size > 150:
        break

cv2.destroyAllWindows()

np.save(f"{name}.npy", np.array(X))
print(data_size,"data_size")
print(f"Data saved. Shape: {np.array(X).shape}")
