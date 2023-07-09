import numpy as np

from transform_3D_utils.utils import calculate_iou


class Tracker:
    def __init__(self, box, id, frame):
        self.boxes = [box]
        self.id = id
        self.frames = [frame]
        self.centers = []
        self.missing = -1

    def get_iou(self, box):
        return calculate_iou(box, self.boxes[-1])

    def assign(self, box, frame):
        self.boxes.append(box)
        self.frames.append(frame)
        self.missing = -1

    def assign_center(self, center):
        self.centers.append(center)

    def check_misses(self, keep):
        self.missing += 1
        return self.missing > keep

    def get_speed(self, projector, fps):
        if len(self.centers) < 3:
            return 5

        centers_world_array = np.array([projector(np.array([p[0], p[1], 1])) for p in self.centers])
        dists = np.linalg.norm(centers_world_array[1:] - centers_world_array[:-1], axis=1)
        frame_diffs = np.array(self.frames)[1:] - np.array(self.frames)[:-1]
        speeds = 3.6 * dists / (frame_diffs / fps)

        return np.median(speeds)
