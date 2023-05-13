import numpy as np
from math import floor

def remove_linearities(points) :
	data = np.array(points)

	data = np.swapaxes(data, 0, 2)[0]

	valid_points = []

	for point in data :
		point = np.swapaxes(point, 0, 1)
		x = point[0]
		y = point[1]
		t = list(range(0, x.shape[0]))
		# print(t)
		# print("Len(t) : ", len(t))
		model, residuals1, rank, singular, thresh = np.polyfit(t, x, 1, full=True)
		model, residuals2, rank, singular, thresh = np.polyfit(t, y, 1, full=True)

		residuals = residuals1+residuals2


		
		if residuals > 1000 : 
			valid_points.append(point)
	
	valid_points = np.array(valid_points)

	print("Linearities removal : ", points.shape[1], '->', valid_points.shape[0])

	if len(valid_points) > 0 :
		valid_points = np.swapaxes(valid_points, 1, 2)
	
	return valid_points

def pool_points(points) :
	all_points = []

	for point in points :
		for pos in point : 
			all_points.append(pos)

	all_points = np.array(all_points)
	all_points = all_points.astype(int)

	return all_points.tolist()

def is_on_edge(point, w, h, threshold=5) :
	if point[0] < threshold :
		return True
	if point[0] > w-threshold :
		return True
	if point[1] < threshold :
		return True
	if point[1] > h-threshold :
		return True
	return False

def remove_edge_effects(points, width, height) :
	points = np.array(points)
	points = np.swapaxes(points, 0, 1)
	new_points = []

	for point_history in points :
		ok = True
		for pos in point_history :
			if is_on_edge(pos[0], width, height) :
				ok = False
		if ok :
			new_points.append(point_history)

	new_points = np.array(new_points)
	print("Edge removal : ", points.shape[0], '->', new_points.shape[0])
	new_points = np.swapaxes(new_points, 0, 1)

	return new_points

def remove_outliers(pooled_points, width, height, size=40) :
	points = pooled_points

	boxes = {}

	cleared_points = []

	for p in points :
		box_w = floor(p[0]/size)
		box_h = floor(p[1]/size)

		#print("Point ", p, " is in box [%d,%d]"%(box_w, box_h))

		if (box_w, box_h) not in boxes : 
			boxes[(box_w, box_h)] = [p]
		else :
			if p not in boxes[(box_w, box_h)] :
				boxes[(box_w, box_h)] += [p]


	count = [len(boxes[k]) for k in boxes]
	count.sort()
	high_mean = sum(count[-15:-1])/10
	threshold = int(high_mean/10)
	n_selected = len([c for c in count if c>threshold])
	print("Outlier removal : %d boxes kept of %d (t=%d, size=%d)"%(n_selected, len(count), threshold, size))

	for k in boxes :
		#print("Box ", k, " : ", len(boxes[k]))
		if len(boxes[k]) > threshold :
			for p in boxes[k] :
				cleared_points.append(p)

	return cleared_points


