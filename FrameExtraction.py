import cv2 as cv
import numpy as np
import logging

def extract_frames(path, time_range, zone=None) : 
	
	start, end = time_range
	frames = list()

	capture = cv.VideoCapture(cv.samples.findFileOrKeep(path))

	capture.set(1, start+1)


	frame_i = start
	while frame_i < end:
		ret, frame = capture.read()
		if frame is None:
			break
		if zone != None :
			frame = frame[zone[1][0]:zone[1][1], zone[0][0]:zone[0][1]]
		frames.append(frame)
		frame_i+=1
	
	capture.release()

	return frames