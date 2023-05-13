from SparseFlow import SparseFlow
from StillRemoval import StillRemoval
from SceneRecorder import *

class Scene :

	path = None
	start = None
	end = None
	zones = None
	zones_frames_ok = []
	display = False
	id = 0

	def __init__(self, path, id, start, end) :
		self.path = path
		self.start = start
		self.end = end
		self.id = id
		self.zones_frames_ok = []
		self.display = False

	def get_range(self) :
		return self.start, self.end

	def __str__(self) :
		return "Scene of '%s' [%d-%d]"%(self.path, self.start, self.end)

	def __repr__(self) :
		return self.__str__()

	def scan_animation(self) :
		flow = SparseFlow(self, display=self.display)
		self.zones = flow.analyze()
		flow.clear()

	def remove_stills(self, zone_id) :
		print("Still removal for #%d.%d"%(self.id, zone_id))
		zone = self.zones[zone_id]
		sr = StillRemoval(self, zone, display=self.display)
		frames_ok = sr.scan_points()
		self.zones_frames_ok.append(frames_ok.copy())
		del sr

	def save_frames(self) :
		for i in range(0, len(self.zones)) :
			record_zone(self, i)

