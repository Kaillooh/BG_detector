import cv2
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev
import math
import argparse
from sklearn.cluster import KMeans

class CameraTrack :

	capture = None
	width = 0
	height = 0
	source = None
	display = False
	point_history = []
	trajectory_analysis = []
	intersection_points = []
	center = None

	def __init__(self, source, display=True) :
		self.source = source

		self.display = display

		self.capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(source))
		
		self.width  = int(self.capture.get(3))
		self.height = int(self.capture.get(4))


	#### DISPLAY ####

	def displayPoints(self, image, points) :
		for point in points : 
			coord = (int(point[0][0]), int(point[0][1]))
			# print(coord)
			image = cv2.circle(image, coord, radius=5, color=(0, 0, 255), thickness=2)
		return image

	#### HISTORY MANAGEMENT ####

	def init_history(self, points) :
		self.point_history = []
		for i in range(0, len(points)) :
			point = (int(points[i][0][0]), int(points[i][0][1]))
			self.point_history.append([point])

	def add_to_history(self, points, status) :
		for i in range(0, len(points)) :
			if status[i] == 1 : 
				point = (int(points[i][0][0]), int(points[i][0][1]))
				self.point_history[i].append(point)
			else :
				self.point_history[i].append(None)
	
	#### MAIN ANALYSIS LOOP ####

	def locatePoints(self) :
		capture = self.capture
		print("Opening video stream : %dx%d"%(self.width, self.height))

		i=0

		base_points = None
		base_frame = None

		while True:
			ret, frame = capture.read()
			if frame is None:
				break

			w,h = frame.shape[:2]
			# print("%dx%d"%(w,h))

			if i == 0 :
				frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

				base_frame = frame_gray

				# Detect feature points in previous frame
				base_points = cv2.goodFeaturesToTrack(frame_gray,
				                                 maxCorners=0,
				                                 qualityLevel=0.001,
				                                 minDistance=70,
				                                 blockSize=3)

				frame = self.displayPoints(frame, base_points)

				self.init_history(base_points)

			else :
				frame2_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 

				frame2_pts, status, err = cv2.calcOpticalFlowPyrLK(base_frame, frame2_gray, base_points, None) 

				assert base_points.shape == frame2_pts.shape 

				idx = np.where(status==1)[0]
				frame2_pts = frame2_pts[idx]

				frame = self.displayPoints(frame, frame2_pts)
				self.add_to_history(frame2_pts, status)

			if self.display :
				cv2.imshow("Points", frame)
				keyboard = cv2.waitKey(30)

			i+=1

	#### TRAJECTORY ANALYSIS ####

	def getPointXY(self, i) :
		x = []
		y = []
		for j in range(0, len(self.point_history[i])) :
			if self.point_history[i][j] != None :
				x.append(self.point_history[i][j][0])
				y.append(self.point_history[i][j][1])

		return (x,y)

	def regression(self, i, show_plot=False) :
		x,y = self.getPointXY(i)
		# print(x)
		# print(y)

		model, residuals, rank, singular, thresh = np.polyfit(x, y, 1, full=True)
		print(model)
		print(residuals)

		if show_plot : 
			plt.scatter(x,y)
			plt.show()

		if residuals.shape[0] != 1 :
			return None
		if model.shape[0] != 2 :
			return None

		return model[0], model[1], residuals[0]

	def displayModel(self, a, b, image) :
		x1 = 0
		x2 = self.width-1
		y1 = int(x1*a+b)
		y2 = int(x2*a+b)

		cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)

		return image

	def calculate_trajectory(self, show_plot=False) :
		
		for i in range(0, len(self.point_history)) :

			reg_result = self.regression(i, show_plot)
			if reg_result == None :
				break
			a,b,r = reg_result

			if r < 500 :
				self.trajectory_analysis.append((a,b))
				


	def show_trajectory(self, timeout=3000) :

		self.capture.set(1, 1)
		ret, image = self.capture.read()

		for a,b in self.trajectory_analysis :
			image = self.displayModel(a, b, image)

		if self.center != None :
			image = cv2.circle(image, self.center, radius=10, color=(0, 0, 255), thickness=2)

		cv2.imshow("Trajectory", image)
		cv2.waitKey(timeout)

	def calculate_intersection(self, n_tries=10, remove_outlier=False) :
		intersect = []

		max_n = min(n_tries, len(self.trajectory_analysis))
		for i in range(0, max_n) :
			points = []
			for j in range(0, max_n) :
				a1, b1 = self.trajectory_analysis[i]
				a2, b2 = self.trajectory_analysis[j]

				xp = (b2-b1)/(a1-a2)
				yp = a1*xp+b1

				print("Results : ", xp, yp)

				if math.isnan(xp) == False :
					points.append([int(xp),int(yp)])
			intersect.append(points)

		self.intersection_points = intersect



	def remove_outlier_trajectory(self) :
		intersect = self.intersection_points
		devs = []
		for i in range(0, len(intersect)) :
			pool = []
			for j in range(0, len(intersect)) :
				if i != j :
					pool += intersect[j]
			# print("Pool : ", pool)

			data = np.array(pool)
			stdev = np.std(data, axis=0)
			print("Stdev without %d : "%i, np.sum(stdev))
			devs.append(np.sum(stdev))
		print(devs)
		min_id = np.argmin(np.array(devs))
		print("Outlier is #%d"%min_id)
		del self.trajectory_analysis[min_id]

	def arbitrate_center(self) :
		full = []
		for points in self.intersection_points :
			full += points
		data = np.array(full)
		center = np.mean(data, axis=0)
		center = tuple(center.astype(int))
		print(center)
		self.center = center







if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--input', '-i', type=str, help='Path to a video or a sequence of image.')
	args = parser.parse_args()

	cam_track = CameraTrack(args.input, display=True)
	cam_track.locatePoints()
	print(cam_track.point_history[0])
	print("Points : ", len(cam_track.point_history))

	cam_track.calculate_trajectory(show_plot=False)
	cam_track.show_trajectory(500)

	n_elim = int(len(cam_track.trajectory_analysis)*0.3)
	print("Eliminating %d : "%n_elim)
	for i in range(0, n_elim) :
		cam_track.calculate_intersection(200)
		cam_track.remove_outlier_trajectory()
		cam_track.show_trajectory(300)

	cam_track.calculate_intersection(200)
	cam_track.arbitrate_center()
	cam_track.show_trajectory(3000)