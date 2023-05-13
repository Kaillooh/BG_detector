from __future__ import print_function
import cv2 as cv
import argparse
import numpy as np
import sys
import tracemalloc
import gc

from SceneDivider import SceneDivider
from SparseFlow import SparseFlow



parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by OpenCV. You can process both videos and images.')

parser.add_argument('--input', '-i', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
parser.add_argument('--range', '-r', type=str, help='Frame range for analysis.', default='')
parser.add_argument('--display', '-d', dest='display', action='store_true')

args = parser.parse_args()

time_range = None
if args.range != "" :
    split = args.range.split(":")
    time_range = [int(split[0]), int(split[1])]

scene_div = SceneDivider(args.input, display=True, resize=0.3)
scene_div.divide(time_range)
scene_div.clear_scenes()
scenes = scene_div.get_scenes()

print('%d scenes detected'%len(scenes))
for scene in scenes : 
    print(scene.get_range())

del scene_div
cv.destroyAllWindows()

for i in range(0,len(scenes)) :

    try :
        scene = scenes[i]

        if args.display :
            scene.display = True

        print("\nAnalyzing scene #%d (%d frames) [%d-%d]"%(scene.id, scene.end-scene.start, scene.start, scene.end))
        scene.scan_animation()
        cv.destroyAllWindows()

        for zone_id in range(0, len(scene.zones)) :
            scene.remove_stills(zone_id)
            cv.destroyAllWindows()

        scene.save_frames()
    
    except :
        print("ERROR during analysis of scene #%d"%i)
        print(sys.exc_info()[0])
        # raise


