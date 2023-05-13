import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
import ZoneSlicer as zs
from FrameExtraction import extract_frames
import gc
import time
import flow_points_analysis as fl


class SparseFlow :
	scene = None
	path = None
	width = 0
	height = 0
	color = (0, 255, 0)
	point_history = []
	frames = []
	display = True

	def __init__(self, scene, zone=None, display=True) :
		self.scene = scene
		self.path = scene.path
		self.frames = list()
		self.display = display
		

		capture = cv.VideoCapture(cv.samples.findFileOrKeep(self.path))

		self.width  = int(capture.get(3))
		self.height = int(capture.get(4))

		self.frames = extract_frames(self.path, self.scene.get_range(), zone=zone)

	### FRAME MANAGEMENT ###

	def resize(self, frame, ratio=2) :
		return cv.resize(frame, (int(self.width/2), int(self.height/2)))


	### DISPLAY ###

	def display_flow(self, points, frame, mask) :
		if self.display == False :
			return

		color = self.color
		for i, (new, old) in enumerate(zip(*points)):
			# Returns a contiguous flattened array as (x, y) coordinates for new point
			a, b = new.ravel()
			# Returns a contiguous flattened array as (x, y) coordinates for old point
			c, d = old.ravel()
			# Draws line between new and old position with green color and 2 thickness
			mask = cv.line(mask, (a, b), (c, d), color, 2)
			# Draws filled circle (thickness of -1) at new position with green color and radius of 3
			frame = cv.circle(frame, (a, b), 3, color, -1)
			# Overlays the optical flow tracks on the original frame
		output = cv.add(frame, mask)

		cv.imshow("sparse optical flow", output)
		cv.waitKey(30)

		return mask	

	def display_valid_points(self) :
		if self.display == False :
			return

		image = self.frames[0].copy()

		for point in self.point_history : 
			image = cv.circle(image, (point[0], point[1]), 3, self.color, -1)

		cv.imshow("final points", image)
		cv.waitKey(2000)

	def display_zones(self, zones) :
		if self.display == False :
			return

		img = self.frames[0].copy()

		for zone in zones :
			segX = zone[0]
			segY = zone[1]
			cv.rectangle(img, (segX[0], segY[0]), (segX[1], segY[1]), (255,0,0), 2)
		
		cv.imshow("final zones", img)
		cv.waitKey(2000)


	### DATA PROCESSING ###

	def group_zones(self) :
		all_points = np.array(self.point_history)

		if all_points.shape[0] == 0 :
			return []

		full_zone = ((0, self.width-1), (0, self.height-1))
		zones = [full_zone]

		zones = zs.slice(all_points, axis=0, zones=zones)
		zones = zs.slice(all_points, axis=1, zones=zones)
		zones = zs.slice(all_points, axis=0, zones=zones)

		zones = zs.dilate(zones, 10, self.width, self.height)

		return zones


	### FRAME ANALYSIS LOOP ###

	def track_points(self, backwards=False, skip=0) :
		feature_params = dict(maxCorners = 400, qualityLevel = 0.07, minDistance = 5, blockSize = 7)
		lk_params = dict(winSize = (15,15), maxLevel = 2, criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

		first_i = 0+skip
		last_i = len(self.frames)-1
		step = 1

		if backwards :
			first_i = len(self.frames)-1-skip
			last_i = 0
			step = -1


		first_frame = self.frames[first_i]

		prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

		prev = cv.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
		if prev is None :
			return []
		if len(prev) == 0 :
			return []

		mask = np.zeros_like(first_frame)

		point_history = []

		i = first_i+step

		while i != last_i+step :
			# print("Analyzing frame #%03d"%i)
			frame = self.frames[i].copy()

			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

			next, status, error = cv.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
			good_old = prev[status == 1]
			good_new = next[status == 1]

			mask = self.display_flow((good_new, good_old), frame, mask)

			prev_gray = gray.copy()
			point_history.append(prev)
			prev = next.reshape(-1, 1, 2)

			i+=step

		return point_history

	### FULL ANALYSIS PROCESS ###

	def analyze(self) :
		self.point_history = []

		print("Full flow analysis of scene #%d."%self.scene.id)

		points1 = self.track_points(False)
		print("Points before lin in full flow : ", np.array(points1).shape)
		points1 = fl.remove_edge_effects(points1, self.width, self.height)
		points1 = fl.remove_linearities(points1)
		points1 = fl.pool_points(points1)
		self.point_history += points1

		points2 = self.track_points(True)
		points2 = fl.remove_edge_effects(points2, self.width, self.height)
		points2 = fl.remove_linearities(points2)
		points2 = fl.pool_points(points2)
		self.point_history += points2

		self.point_history = fl.remove_outliers(self.point_history, self.width, self.height)

		self.display_valid_points()

		zones = self.group_zones()
		self.display_zones(zones)
		return zones

	### CLOSING ###

	def clear(self) :
		# print("Referers to self.frames : ", gc.get_referrers(self.frames))
		# print("Referers to self.frames[0]", gc.get_referrers(self.frames[0]))
		del self.frames
		gc.collect()
		time.sleep(1)
		# print("Frame released")