from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import os
from Scene import Scene
import progressbar

class SceneDivider :

	path = None
	capture = None
	resize_ratio = 1

	scenes = []

	width = 0
	height = 0

	display = None


	def __init__(self, path, display=False, resize=1) :
		self.path = path
		self.open_capture()
		self.display = display
		self.resize_ratio = resize

	def open_capture(self) :

		self.capture = cv.VideoCapture(cv.samples.findFileOrKeep(self.path))

		if not self.capture.isOpened:
			print('Unable to open: ' + self.path)
			exit(0)

		self.width  = int(self.capture.get(3))
		self.height = int(self.capture.get(4))

	def get_scenes(self) :
		return self.scenes

	def clear_scenes(self) :
		new_scenes = []

		for scene in self.scenes :
			start,end = scene.get_range()
			if end - start > 10 :
				new_scenes.append(scene)

		self.scenes = new_scenes

	def divide(self, time_range=None) :		

		backSub = cv.createBackgroundSubtractorMOG2()

		capture = self.capture

		sumImg = -1

		i=0
		scene_start = 0

		if time_range != None :
			capture.set(1, time_range[0])
			i=time_range[0]
			scene_start = time_range[0]

		last_frame = None

		zone_index = {}

		scene_id = 0

		length = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
		if time_range != None :
			length = time_range[1]-time_range[0]
		print("Sequence mapping : ")
		bar = progressbar.ProgressBar(maxval=length, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
		bar.start()

		while True:
			ret, frame = capture.read()
			if frame is None:
				break

			# print("#%04d"%i)

			if time_range is None :
				bar.update(i)
			else :
				bar.update(i-time_range[0])


			frame = cv.resize(frame, (int(self.width*self.resize_ratio), int(self.height*self.resize_ratio)))
			
			fgMask = backSub.apply(frame)

			
			cv.rectangle(frame, (10, 2), (100,20), (255,255,255), -1)
			cv.putText(frame, str(capture.get(cv.CAP_PROP_POS_FRAMES)), (15, 15),
					   cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
			
			
			fgMask = cv.threshold(fgMask, 254, 255, cv.THRESH_BINARY)[1]

			change_ratio = fgMask.sum()/(self.width*self.height*255*self.resize_ratio*self.resize_ratio)

			# print("Change ratio at %d : "%(i), change_ratio)

			if change_ratio > 0.4 :
				backSub = cv.createBackgroundSubtractorMOG2()

				# print("NOUVELLE SCENE")
				new_scene = Scene(self.path, scene_id, scene_start, i-1)
				scene_id += 1
				self.scenes.append(new_scene)

				if type(sumImg) is int :
					sumImg = fgMask
				sumImg[True] = 0
				scene_start = i


			elif self.display :
				cv.imshow('Mvt', sumImg)

				# cv.imshow('Mvt', fgMask)

			if isinstance(sumImg, np.ndarray) == False : 
				sumImg = fgMask
			elif change_ratio <= 0.6 :
				sumImg = np.add(fgMask, sumImg)

			
			keyboard = cv.waitKey(1)
			if keyboard == 'q' or keyboard == 27:
				break

			last_frame = frame

			i+=1

			if time_range != None :
				if i > time_range[1] :
					break

		if i-1-scene_start > 5 :
			new_scene = Scene(self.path, scene_id, scene_start, i-1)
			self.scenes.append(new_scene)

		bar.finish()