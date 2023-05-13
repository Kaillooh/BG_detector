import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import argparse


class Flow :
	path = None
	capture = None
	width = 0
	height = 0

	def __init__(self, path) :
		self.path = path
		self.capture = cv.VideoCapture(cv.samples.findFileOrKeep(path))

		self.width  = int(self.capture.get(3))
		self.height = int(self.capture.get(4))


	def display_flow(self, flow, frame) :
		mask = np.zeros_like(frame)
		mask[..., 1] = 255
		magnitude, angle = cv.cartToPolar(flow[..., 0], flow[..., 1])
		# Sets image hue according to the optical flow direction
		mask[..., 0] = angle * 180 / np.pi / 2
		# Sets image value according to the optical flow magnitude (normalized)
		mask[..., 2] = cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX)
		# Converts HSV to RGB (BGR) color representation
		rgb = cv.cvtColor(mask, cv.COLOR_HSV2BGR)
		# Opens a new window and displays the output frame
		cv.imshow("dense optical flow", rgb)
		cv.waitKey(5)

	def resize(self, frame, ratio=2) :
		return cv.resize(frame, (int(self.width/2), int(self.height/2)))

	def analyze(self) :
		capture = self.capture

		ret, first_frame = capture.read()
		first_frame = self.resize(first_frame)
		prev_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)

		while True:
			ret, frame = capture.read()
			frame = self.resize(frame)
			if frame is None:
				break

			gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

			flow = cv.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

			self.display_flow(flow, frame)

			prev_gray = gray

if __name__ == "__main__" :
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--input', '-i', type=str, help='Path to a video or a sequence of image.')
	args = parser.parse_args()

	flow_parser = Flow(args.input)
	flow_parser.analyze()