# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:26:56 2021

@author: shambhu
"""

MIN_CONF = 0.3
NMS_THRESH = 0.3
USE_GPU = False
MIN_DISTANCE = 50
output = 'social_distance'

import numpy as np
import cv2

# frame = image, dnet =  darknet backbone feature extractor with yolov3, ln = layer names.

def detect_people(frame, dnet, ln, personIdx=0):
	(H, W) = frame.shape[:2]
	results = []
    # for a balanced model we use 416x416: size of image as input
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	dnet.setInput(blob)    # input to dark net as blob#
	layerOutputs = dnet.forward(ln)
	boxes = []
	centroids = []
	confidences = []
    
	for output in layerOutputs:
		for detection in output:
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]
			if classID == personIdx and confidence > MIN_CONF:
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))
				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
                
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONF, NMS_THRESH) #non max suppression for boxes
	if len(idxs) > 0:
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			r = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(r)
	return results

from scipy.spatial import distance as dist
import imutils


LABELS = open('coco.names').read().strip().split("\n") # read labels from coco.names
dnet = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights') # get yolo weights

if USE_GPU:
	dnet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
	dnet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
ln = dnet.getLayerNames()
ln = [ln[i[0] - 1] for i in dnet.getUnconnectedOutLayers()]

vs = cv2.VideoCapture('People Walking.mp4')  # load the input video
writer = None

while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	frame = imutils.resize(frame, width=700) # Resize frame then detect only people in it
	results = detect_people(frame, dnet, ln,
		personIdx=LABELS.index("person"))
	violate = set()

	if len(results) >= 2:# to ensure min 2 people are there
		centroids = np.array([r[2] for r in results])
		D = dist.cdist(centroids, centroids, metric="euclidean")
		for i in range(0, D.shape[0]):
			for j in range(i + 1, D.shape[1]):
				if D[i, j] < 50:
					violate.add(i)
					violate.add(j)

	for (i, (prob, bbox, centroid)) in enumerate(results):
		(startX, startY, endX, endY) = bbox
		(cX, cY) = centroid
		color = (0, 255, 0)
		if i in violate:
			color = (0, 0, 255)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		cv2.circle(frame, (cX, cY), 5, color, 1)
	text = "Social Distancing Violations: {}".format(len(violate))
	cv2.putText(frame, text, (10, frame.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
    
#     cv2.imshow("Frame", frame)
#  	key = cv2.waitKey(1) & 0xFF
#  	if key == ord("q"):
# 		break

	if output != "" and writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter('output.avi', fourcc, 25,(frame.shape[1], frame.shape[0]), True)
	if writer is not None:
		writer.write(frame)# write the output in disk as file name:'output.avi'
        