import numpy as np



def get_series(points, zone, axis) :
	select = []

	for point in points : 
		if point[0] > zone[0][0] and point[0] < zone[0][1] :
			if point[1] > zone[1][0] and point[1] < zone[1][1] :
				select.append(point)

	# print(select)
	select = np.array(select)
	select = np.swapaxes(select, 0, 1)
	series = select[axis]
	series = np.unique(series)
	series = series.tolist()
	series.sort()
	# print(series)
	return series

def slice(points, axis, zones, threshold=100) :
	new_zones = []
	for zone in zones : 
		series = get_series(points, zone, axis)
		fills = slice_series(series, threshold)

		for fill in fills :
			if axis == 0 :
				new_zones.append((fill, zone[1]))

			if axis == 1 :
				new_zones.append((zone[0], fill))

	return new_zones

def slice_series(series, threshold=100) :
	prev = series[0]
	gaps = []

	for el in series :
		diff = el - prev
		if diff > threshold :
			gaps.append((prev, el))
		prev = el

	# print(gaps)

	zones = gaps_to_fill(gaps, series[0], series[-1])
	
	return zones


def gaps_to_fill(gaps, first, last, threshold=30) :
	prev = first
	fill = []

	for gap in gaps :
		if gap[0]-prev > threshold :
			fill.append((prev, gap[0]))
		prev = gap[1]

	if last-prev > threshold :
		fill.append((prev, last))

	return fill


def dilate(zones, n, width, height) :
	new_zones = []
	for zone in zones :
		a00 = max(zone[0][0]-n,0)
		a01 = min(zone[0][1]+n,width-1)
		a10 = max(zone[1][0]-n,0)
		a11 = min(zone[1][1]+n,height-1)

		dilated = ((a00,a01), (a10,a11))

		new_zones.append(dilated)
	return new_zones