# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2
import time
import numpy as np


fd=cv2.CascadeClassifier(r"C:\Users\user1\Desktop\aiml\haarcascade_frontalface_default.xml")
fdpf=cv2.CascadeClassifier(r"C:\Users\user1\Desktop\aiml\haarcascade_profileface.xml")
SENSITIVITY = 1.6
CALIBRATION_SAMPLE_RATE = 100

    










MODE = "COCO"

if MODE is "COCO":
    protoFile = r"C:\Users\user1\Desktop\pose\pose_deploy_linevec.prototxt"
    weightsFile =r"C:\Users\user1\Desktop\pose\pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


inWidth = 368
inHeight = 368
threshold = 0.1


#input_source = "sample_video.mp4"
cap = cv2.VideoCapture(0)
hasFrame, frame = cap.read()

#vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

while hasFrame:
    t = time.time()

    
    
    hasFrame, frame = cap.read()
    
    
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv2.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            print(x,y)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frameCopy, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(frameCopy, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(frameCopy, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            #print(partA,partB)

    cv2.putText(frameCopy, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv2.FONT_HERSHEY_COMPLEX, .8, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.putText(frame, "OpenPose using OpenCV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 50, 0), 2, lineType=cv2.LINE_AA)
    # cv2.imshow('Output-Keypoints', frameCopy)
    cv2.imshow('Output-Skeleton', frameCopy)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=fd.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2,minSize=(100, 100),flags=0)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        print(w)
        if w > CALIBRATION_SAMPLE_RATE * SENSITIVITY:
            print('please sit straight');
        else:
            print('correct posture');
    faces1=fdpf.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=2,minSize=(100, 100),flags=0)
    for (x,y,w,h) in faces1:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        print(w)
        if w > CALIBRATION_SAMPLE_RATE * SENSITIVITY:
            print('please sit straight'); 
        else:
            print('correct posture');
    cv2.imshow('img',frame)
    cv2.imshow('img',frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):# Hit `q` to exit
        break

cap.release()
#cv2.imshow('img',a)q
cv2.waitKey(0) 

cv2.destroyAllWindows()