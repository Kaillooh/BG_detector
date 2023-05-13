from FrameExtraction import *
import os

def record_zone(scene, zone_id) :
	zone = scene.zones[zone_id]
	frames = extract_frames(scene.path, scene.get_range(), zone)
	frames_ok = scene.zones_frames_ok[zone_id]

	short_path = scene.path.split("/")[-1]

	folder = "frames/%s/%d.%d/"%(short_path, scene.id, zone_id)
	if not os.path.exists(folder) :
		os.makedirs(folder)
	print("Storing zone #%d.%d at '%s'"%(scene.id, zone_id, folder))

	print("Recording %d frames for #%d.%d"%(len(frames_ok), scene.id, zone_id))

	for i in frames_ok :

		still = frames[i]
		impath = '%sframe%03d.jpg'%(folder, i)
		cv.imwrite(impath, still)