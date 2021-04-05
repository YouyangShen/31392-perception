# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 13:21:46 2021

@author: Youyang Shen
"""

import cv2
import numpy as np


def update(x, P, Z, H, R):
    ### Insert update function
    y = Z - H @ x
    S = H @ P @ np.transpose(H) + R
    K = P @ np.transpose(H) @ np.linalg.pinv(S)
    x_update = x + K @ y
    P_update = (I - K @ H) @ P
    return x_update, P_update

def predict(x, P, F, u):
    ### insert predict function
    x_predict = F @ x + u
    P_predict = F @ P @ np.transpose(F)
    return x_predict, P_predict   
    
    
### Initialize Kalman filter ###
# The initial state (6x1).
X = np.array([[0], # Position along the x-axis
              [0], # Velocity along the x-axis
              [0], # Acc along the x-axis
              [0], # Position along the y-axis
              [0], # Velocity along the y-axis
              [0]])# Acc along the y-axis

# The initial uncertainty (6x6).
P = np.identity(6)*20000

# The external motion (6x1).
u = np.array([[0],
              [0],
              [0],
              [0],
              [0],
              [0]])

# The transition matrix (6x6). 
F = np.array([[1, 1, 1/2, 0, 0, 0],
              [0, 1, 1, 0, 0, 0],
              [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 1, 1, 1/2],
              [0, 0, 0, 0, 1, 1],
              [0, 0, 0, 0, 0, 1]])

# The observation matrix (2x6).
H = np.array([[1, 0, 0, 0, 0, 0],[0,0,0,1,0,0]])

# The measurement uncertainty.
R = np.array([[1],[1]])

I = np.identity(6)

# Load the video
cap = cv2.VideoCapture('rolling_ball.mp4')
if not cap.isOpened():
    print("Cannot open video")
    exit()

# Looping through all the frames
while True:
    ret, frame = cap.read()
    if not ret:
        break    
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    mask = cv2.inRange(frame, np.array([10, 10, 95]), np.array([80, 80, 255])) # cv2.inRange(rgb, lower_red, upper_red)
    gray = gray * mask
    # Canny edge detection
    edges = cv2.Canny(gray, 50, 60)
    # Hough transform
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 50, param1=60, param2=20, minRadius=65, maxRadius=80)
    
    ### If the ball is found, update the Kalman filter ###
    if circles is not None:
        circles = circles[0]
        for c in range(len(circles)):
            x = np.int(circles[c][0])
            y = np.int(circles[c][1])
        cv2.circle(frame, (x, y), 80, (0, 0, 255), 5)
        z = np.array([[x],[y]]) 
        X, P = update(X, P, z, H, R)
    else:
        cv2.circle(frame, (np.int(X[0]), np.int(X[3])), 80, (0, 0, 255), 5)
    
    ### Predict the next state
    X, P = predict(X, P, F, u)
    
    ### Draw the current tracked state and the predicted state on the image frame ###
    cv2.circle(frame, (np.int(X[0]), np.int(X[3])), 80, (255, 0, 0), 5)
    
    # Show the frame
    cv2.namedWindow('Frame',0)
    cv2.resizeWindow('Frame',800, 500)
    cv2.imshow('Frame', frame)
    cv2.waitKey(50)

    
cap.release()
cv2.destroyAllWindows()