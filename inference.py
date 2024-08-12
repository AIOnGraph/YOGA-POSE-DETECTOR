
import cv2
from sample_utils import get_ice_servers
import numpy as np
import mediapipe as mp
from keras.models import load_model
import av
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode

import streamlit as st

def inFrame(lst):
    return (
        lst[28].visibility > 0.6 and lst[27].visibility > 0.6 and 
        lst[15].visibility > 0.6 and lst[16].visibility > 0.6
    )

model = load_model("model.h5")
label = np.load("labels.npy")

holistic = mp.solutions.pose
holis = holistic.Pose()
drawing = mp.solutions.drawing_utils

class VideoProcessor1(VideoProcessorBase):
    def __init__(self):
        self.model = model
        self.label = label
        self.holis = holis
        self.drawing = drawing

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        lst = []
        # window = np.zeros((1080, 1080, 3), dtype="uint8")
        # window = np.zeros((940, 940, 3), dtype="uint8")
        img = cv2.flip(img, 1)
        
        res = self.holis.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # img = cv2.blur(img, (4, 4))
        
        if res.pose_landmarks and inFrame(res.pose_landmarks.landmark):
            for i in res.pose_landmarks.landmark:
                lst.append(i.x - res.pose_landmarks.landmark[0].x)
                lst.append(i.y - res.pose_landmarks.landmark[0].y)

            lst = np.array(lst).reshape(1, -1)
            p = self.model.predict(lst)
            pred = self.label[np.argmax(p)]

            if p[0][np.argmax(p)] > 0.75:
                cv2.putText(img, pred, (100, 450), cv2.FONT_ITALIC, 1.3, (0, 255, 0), 2)
            else:
                cv2.putText(img, "Asana is either wrong or not trained", 
                            (100, 180), cv2.FONT_ITALIC, 1.8, (0, 0, 255), 3)

        else:
            cv2.putText(img, "Make Sure Full body visible", (100, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

        self.drawing.draw_landmarks(img, res.pose_landmarks, holistic.POSE_CONNECTIONS,
                                    connection_drawing_spec=self.drawing.DrawingSpec(
                                        color=(255, 255, 255), thickness=6),
                                    landmark_drawing_spec=self.drawing.DrawingSpec(
                                        color=(0, 0, 255), circle_radius=3, thickness=3))

        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("YOGA POSE DETECTION")

webrtc_streamer(key="sample",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration={
            "iceServers": get_ice_servers(),
            "iceTransportPolicy": "relay",
        },
        media_stream_constraints={
                        "video": True,
                        "audio": False,
                    },
        video_processor_factory=lambda: VideoProcessor1(),
        async_processing=True
        )
    

