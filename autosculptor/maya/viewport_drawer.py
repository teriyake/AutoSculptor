import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.api.OpenMayaRender as omr
from maya.api.OpenMayaRender import MPxDrawOverride
import maya.OpenMayaMPx as omp
from autosculptor.core.data_structures import Stroke


class ViewportDataCache:
	"""Singleton storage for viewport suggestions"""

	__instance = None

	def __new__(cls):
		if cls.__instance is None:
			cls.__instance = super(ViewportDataCache, cls).__new__(cls)
			cls.__instance.suggestions = []
			cls.__instance.mesh_transform = om.MTransformationMatrix()
		return cls.__instance

	def update_suggestions(self, suggestions):
		self.suggestions = suggestions

	def get_suggestions(self):
		return self.suggestions

	def set_mesh_transform(self, matrix):
		self.mesh_matrix = om2.MMatrix(matrix)

	def get_mesh_transform(self):
		return self.mesh_matrix


class SuggestionDrawer(MPxDrawOverride):
	nodeName = "autoSculptorDrawOverride"
	nodeId = om.MTypeId(0x8724)
	drawDBClassification = "drawdb/geometry/autoSculptor"
	drawRegistrantId = "AutoSculptorPlugin"

	def __init__(self, obj):
		super(SuggestionDrawer, self).__init__(obj, None, False)
		self.data_cache = ViewportDataCache()
		self._color = om.MColor(1.0, 0.0, 0.0, 1.0)
		self._line_width = 3
		print("## DEBUG: SuggestionDrawer initialized")

	@classmethod
	def creator(cls):
		return omp.asMPxPtr(cls(om.MObject()))

	@classmethod
	def initialize(cls):
		cls.drawDBClassification = "drawdb/geometry/autoSculptor"
		cls.drawRegistrantId = "AutoSculptorPlugin"

	def supportedDrawAPIs(self):
		return omr.MRenderer.kAllDevices

	def prepareForDraw(self, obj_path, camera_path, frame_context, old_data):
		print("## DEBUG: prepareForDraw called")
		try:
			selection_list = om.MSelectionList()
			selection_list.add(obj_path.partialPathName())
			dag_path = selection_list.getDagPath(0)
			transform_fn = om.MFnTransform(dag_path)
			self.data_cache.set_mesh_transform(transform_fn.transformation())

		except Exception as e:
			print(f"Error in prepareForDraw: {e}")
			import traceback

			traceback.print_exc()

	def hasUIDrawables(self):
		print("## DEBUG: hasUIDrawables called")
		return True

	def addUIDrawables(self, obj_path, draw_manager, frame_context, data):
		print("## DEBUG: Entering addUIDrawables")
		try:
			draw_manager.beginDrawable()

			draw_manager.setColor(om.MColor((1, 1, 0, 1)))
			draw_manager.setLineWidth(3)
			draw_manager.setLineStyle(omr.MUIDrawManager.kSolid)

			suggestions = self.data_cache.get_suggestions()
			print(f"## DEBUG: Number of suggestions to draw: {len(suggestions)}")
			if not suggestions:
				print("## DEBUG: No suggestions to draw (empty list)")
				return

			mesh_matrix = self.data_cache.get_mesh_transform()
			if mesh_matrix is None:
				print("## DEBUG: Mesh transform matrix is None!")
				return

			for stroke in suggestions:
				if not stroke.samples:
					print("## DEBUG: Skipping stroke with no samples")
					continue

				points = om2.MPointArray()
				print(f"## DEBUG: Drawing stroke with {len(stroke.samples)} samples")
				for sample in stroke.samples:
					if sample.position is None:
						print("## DEBUG: Sample position is None!")
						continue
					raw_point = om2.MPoint(sample.position)
					transformed_point = raw_point * mesh_matrix
					points.append(transformed_point)

				if points.length() > 1:
					draw_manager.polyLine(points, closed=False)
					print("## DEBUG: Drew polyline for a suggestion stroke")
				else:
					print(
						"## DEBUG: Not enough points to draw polyline in suggestion stroke"
					)

			draw_manager.endDrawable()
			print("## DEBUG: Finished addUIDrawables without errors (hopefully!)")

		except Exception as e:
			print(f"## DEBUG: ERROR in addUIDrawables: {e}")
			import traceback

			traceback.print_exc()
