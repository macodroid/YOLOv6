import json
import math
import os.path

import cv2
import numpy as np

from tracker import Tracker
from utils import intersection, line

font = cv2.FONT_HERSHEY_SIMPLEX


class Radar:
    def __init__(self,
                 transform_matrix: np.ndarray,
                 inv_transform_matrix: np.ndarray,
                 vanishing_points: list[np.ndarray],
                 vp0_t: np.ndarray,
                 image_size: tuple,
                 projector,
                 video_fps: int,
                 result_path: str,
                 result_name: str,
                 camera_calib_structure: dict,
                 save_often=True,
                 keep=5):
        self.save_often = save_often
        self.transform_matrix = transform_matrix
        self.inv_transform_matrix = inv_transform_matrix
        self.vp1, self.vp2, self.vp3 = vanishing_points
        self.vp0_t = vp0_t
        self.image_size = image_size
        self.keep = keep
        self.projector = projector
        self.video_fps = video_fps
        self.write_path = os.path.join(result_path, "system_" + result_name + '.json')

        self.trackers: list[Tracker] = []
        self.last_id = 0
        self.frame = 0
        # with open("", 'r+') as file:
        #     structure = json.load(file)
        # self.dubska_cars = structure['cars']
        self.json_structure = {'cars': [], 'camera_calibration': camera_calib_structure}

    def process_frame(self, boxes_2d, fubs, frame):
        image = np.copy(frame)
        self.frame += 1
        if self.frame % 5000 == 0 and self.save_often:
            self.write_record()

        for box_2d, fub in zip(boxes_2d, fubs):
            tracker = self.get_tracker(box_2d)
            image, center = self.draw_3d_bbox(tracker, box_2d, fub[0], image)
        self.remove_tracker()
        return image

    def process_frame_offline(self, boxes_2d, fubs):
        self.frame += 1
        if self.frame % 5000 == 0:
            self.write_record()

        for box in boxes_2d:
            _, center = self.get_3d_bbox(boxes_2d, fubs)
            if center is not None:
                track = self.get_tracker(box)
                track.assign_center(center)
        self.remove_tracker()

    def write_record(self):
        with open(self.write_path, 'w') as file:
            json.dump(self.json_structure, file)

    def remove_tracker(self):
        for i in reversed([i for (i, t) in enumerate(self.trackers) if t.check_misses(self.keep)]):
            self.add_record(self.trackers[i])
            del self.trackers[i]

    def add_record(self, track):
        if len(track.frames) < 5:
            return
        frames = []
        posX = []
        posY = []
        for frame, center, box in zip(track.frames, track.centers, track.boxes):
            posX.append(float(center[0]))
            posY.append(float(center[1]))
            frames.append(frame)

        if len(frames) < 5:
            return

        dist = math.sqrt(math.pow(posX[0] - posX[-1], 2) + math.pow(posY[0] - posY[-1], 2))
        if dist > 30:
            entry = {'frames': track.frames, 'id': track.id, 'posX': posX, 'posY': posY}
            self.json_structure['cars'].append(entry)

    def draw_3d_bbox(self, tracker, bbox_2d, fub, img):
        bb_tt, center = self.get_3d_bbox(bbox_2d, fub)

        tracker.assign_center(center)

        bb_tt = [tuple(map(int, point)) for point in bb_tt]

        img = cv2.line(img, bb_tt[0], bb_tt[1], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[1], bb_tt[2], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[2], bb_tt[3], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[3], bb_tt[0], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[0], bb_tt[4], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[1], bb_tt[5], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[2], bb_tt[6], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[3], bb_tt[7], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[4], bb_tt[5], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[5], bb_tt[6], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[6], bb_tt[7], (255, 128, 0), 2)
        img = cv2.line(img, bb_tt[7], bb_tt[4], (255, 128, 0), 2)

        # id = tracker.id
        # speed = tracker.get_speed(self.projector, self.video_fps)
        # img = cv2.putText(img, '{}:{:.2f}'.format(id, speed), bb_tt[0], font, 1, (0, 0, 255), 2, cv2.LINE_AA)

        img = cv2.circle(img, (int(center[0]), int(center[1])), 5, (0, 255, 255), 5)

        return img, center

    def get_3d_bbox(self, bbox_2d: np.ndarray, fub: float):
        x_min = bbox_2d[0]
        y_min = bbox_2d[1]
        x_max = bbox_2d[2]
        y_max = bbox_2d[3]

        cy_0 = fub * (y_max - y_min) + y_min
        bb_t = []
        if self.vp0_t[1] < y_min:
            if (x_min < self.vp0_t[0]) and (self.vp0_t[0] < x_max):
                # print("Case 1")
                cy = cy_0
                cx = x_min
                bb_t.append([cx, cy])
                bb_t.append([x_max, cy])
                bb_t.append([x_max, y_max])
                bb_t.append([cx, y_max])
                tx, ty = intersection(line([cx, cy], self.vp0_t), line([x_min, y_min], [x_min + 1, y_min]))
                bb_t.append([tx, y_min])
                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.inv_transform_matrix)
                bb_tt = [point[0] for point in bb_tt]

                center = (bb_tt[3] + bb_tt[2]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp1), line(bb_tt[4], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp1), line(bb_tt[5], self.vp3)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp1), line(bb_tt[6], self.vp2)))

            elif self.vp0_t[0] < x_min:
                # print("Case 2")
                line1 = line([x_min, y_min], self.vp0_t)
                line2 = line([0, cy_0], [1, cy_0])
                cx, cy = intersection(line1, line2)
                bb_t.append([cx, cy])
                bb_t.append([x_max, cy])
                bb_t.append([x_max, y_max])
                bb_t.append([cx, y_max])
                bb_t.append([x_min, y_min])

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.inv_transform_matrix)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[3] + bb_tt[2]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp1), line(bb_tt[4], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp1), line(bb_tt[5], self.vp3)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp1), line(bb_tt[6], self.vp2)))
            else:  # self.vp0_t[0] > x_max
                # print("Case 3")
                cx, cy = intersection(line([x_max, y_min], self.vp0_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([cx, cy])
                bb_t.append([cx, y_max])
                bb_t.append([x_min, y_max])
                bb_t.append([x_min, cy])
                bb_t.append([x_max, y_min])

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.inv_transform_matrix)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[1] + bb_tt[2]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp1), line(bb_tt[4], self.vp3)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp1), line(bb_tt[5], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp1), line(bb_tt[6], self.vp3)))
        # elif (self.vp0_t[1] > y_max):
        else:
            if (x_min < self.vp0_t[0]) and (self.vp0_t[0] < x_max):
                # print("Case 4")
                cy = cy_0
                bb_t.append([x_min, cy])
                bb_t.append([x_max, cy])
                bb_t.append(intersection(line(self.vp0_t, [x_max, cy]), line([0, y_max], [1, y_max])))
                bb_t.append(intersection(line(self.vp0_t, [x_min, cy]), line([0, y_max], [1, y_max])))
                bb_t.append([x_min, y_min])
                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.inv_transform_matrix)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[2] + bb_tt[3]) / 2

                bb_tt.append(intersection(line(bb_tt[1], self.vp3), line(bb_tt[4], self.vp2)))
                bb_tt.append(intersection(line(bb_tt[2], self.vp3), line(bb_tt[5], self.vp1)))
                bb_tt.append(intersection(line(bb_tt[3], self.vp3), line(bb_tt[4], self.vp1)))

            elif self.vp0_t[0] < x_min:
                # print("Case 5")
                cx, cy = intersection(line([x_min, y_max], self.vp0_t), line([0, cy_0], [1, cy_0]))
                bb_t.append([cx, cy])
                bb_t.append([x_max, cy])
                bb_t.append(list(intersection(line([x_max, cy], self.vp0_t), line([x_min, y_max], [x_max, y_max]))))
                bb_t.append([x_min, y_max])

                bb_t.append([cx, y_min])
                bb_t.append([x_max, y_min])
                bb_t.append(
                    list(intersection(line(bb_t[2], [bb_t[2][0], bb_t[2][1] + 1]), line(self.vp0_t, [x_max, y_min]))))
                bb_t.append(
                    list(intersection(line(bb_t[6], [bb_t[6][0] + 1, bb_t[6][1]]), line([x_min, 0], [x_min, 1]))))

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.inv_transform_matrix)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[3] + bb_tt[2]) / 2
            else:
                # print("Case 6: {}".format(self.vp0_t))
                print("x_max: {}, x_min: {}, y_min:{}, y_max:{}, c:{}".format(x_max, x_min, y_min, y_max, cy_0))
                cx, cy = intersection(line([x_max, y_max], self.vp0_t), line([0, cy_0], [1, cy_0]))

                bb_t.append([x_min, cy])
                bb_t.append([cx, cy])
                bb_t.append([x_max, y_max])
                bb_t.append(list(intersection(line([x_min, cy], self.vp0_t), line([x_min, y_max], [x_max, y_max]))))

                bb_t.append([x_min, y_min])
                bb_t.append([cx, y_min])
                bb_t.append(list(intersection(line([x_max, y_min], [x_max, y_max]), line(self.vp0_t, bb_t[5]))))
                bb_t.append(list(intersection(line(bb_t[6], [bb_t[6][0] + 1, bb_t[6][1]]), line(self.vp0_t, bb_t[4]))))

                bb_t_array = np.array([[point] for point in bb_t], np.float32)
                bb_tt = cv2.perspectiveTransform(bb_t_array, self.inv_transform_matrix)
                bb_tt = [point[0] for point in bb_tt]
                center = (bb_tt[2] + bb_tt[3]) / 2
        return bb_tt, center

    def get_tracker(self, box):
        if len(self.trackers) == 0:
            self.last_id += 1
            new_tracker = Tracker(box, self.last_id, self.frame)
            self.trackers.append(new_tracker)
            return new_tracker

        max_iou = 0.1
        max_tracker = None
        for tracker in self.trackers:
            if tracker.missing == -1:
                continue
            iou = tracker.get_iou(box)
            if iou > max_iou:
                max_iou = iou
                max_tracker = tracker
        if max_tracker is None:
            self.last_id += 1
            new_tracker = Tracker(box, self.last_id, self.frame)
            self.trackers.append(new_tracker)
            return new_tracker
        else:
            max_tracker.assign(box, self.frame)
            return max_tracker

    # def dubska_point(self, image_b):
    #     for car in self.dubska_cars:
    #         try:
    #             idx = car["frames"].index(self.frame)
    #             posX = car["posX"][idx]
    #             posY = car["posY"][idx]
    #             image_b = cv2.circle(image_b, (int(posX), int(posY)), 5, (255, 0, 0), 3)
    #         except ValueError:
    #             pass
    #     return image_b
