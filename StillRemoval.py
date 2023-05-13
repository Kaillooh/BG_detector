import cv2 as cv
from FrameExtraction import extract_frames
from SparseFlow import SparseFlow
import numpy as np
import gc

import flow_points_analysis as fl

class StillRemoval :

	scene = None
	zone = None
	frames = None
	start = 0
	end = 0
	display = True

	def __init__(self, scene, zone, display=True) :
		self.scene = scene
		self.zone = zone
		self.start, self.end = self.scene.get_range()
		self.frames = extract_frames(self.scene.path, self.scene.get_range(), zone)
		self.display = display

	def play(self) :
		for frame in self.frames :
			cv.imshow("zone", frame)
			cv.waitKey(10)

	
	def scan_points(self) :
		frames_ok = [0]
		flow = SparseFlow(self.scene, zone=self.zone, display=False)
		points = flow.track_points(False)


		points = np.array(points)


		change_history = []

		for step in range(1, points.shape[0]) :
			change_sum = 0
			for i in range(0, points.shape[1]) :
				t1 = points[step][i][0]
				t0 = points[step-1][i][0]
				diff = (t1[0]-t0[0])**2 + (t1[1]-t0[1])**2 
				change_sum += diff
			change_sum = int(change_sum/points.shape[1])
			change_history.append(change_sum)

			# print("Change : ", change_sum)
			
		
		values = change_history.copy()
		values.sort()
		n = len(values)
		values = values[-1-int(n/4):-1]
		mean = int(sum(values)/len(values))
		# print("MEAN : ", mean)
		threshold = int(mean/50)
		threshold = max(threshold, 1)
		# print("THRESHOLD : ", threshold)

		for step in range(1, points.shape[0]) :
			change_sum = change_history[step-1]
			if change_sum > threshold :
				if self.display :
					cv.imshow("select", self.frames[step])
					cv.waitKey(100)
				frames_ok.append(step)


		flow.clear()
		del flow
		gc.collect()

		print("%d frames kept of %d"%(len(frames_ok), points.shape[0]))
		
		return frames_ok
