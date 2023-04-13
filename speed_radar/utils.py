import json
import math
from typing import Tuple

import cv2
import numpy as np


def get_calibration_params(calib_file) -> dict:
    with open(calib_file, 'r+') as f:
        calibration = json.load(f)
    return calibration['camera_calibration']


def get_world_coordinates_on_road_plane(p, focal, road_plane, pp):
    p = p / p[2]
    pp = pp / pp[2]
    pp_w = np.concatenate((pp[0:2], [0]))
    p_w = np.concatenate((p[0:2], [focal]))
    dir_vec = p_w - pp_w
    t = -np.dot(road_plane, np.concatenate((pp_w, [1]))) / np.dot(road_plane[0:3], dir_vec)
    return pp_w + t * dir_vec


def get_focal_length(vp1, vp2, pp):
    return math.sqrt(-np.dot(vp1[0:2] - pp[0:2], vp2[0:2] - pp[0:2]))


def compute_camera_calibration(
        vanishing_points: list[list], perspective_point: np.array
):
    focal = get_focal_length(
        np.asarray(vanishing_points[0]),
        np.asarray(vanishing_points[1]),
        perspective_point,
    )
    vp1 = np.concatenate((vanishing_points[0], [1]))
    vp2 = np.concatenate((vanishing_points[1], [1]))
    pp = np.concatenate((perspective_point, [1]))
    vp1W = np.concatenate((vanishing_points[0], [focal]))
    vp2W = np.concatenate((vanishing_points[1], [focal]))
    ppW = np.concatenate((perspective_point, [0]))
    vp3W = np.cross(vp1W - ppW, vp2W - ppW)
    vp3 = np.concatenate((vp3W[0:2] / vp3W[2] * focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal])) - ppW
    roadPlane = np.concatenate((vp3Direction / np.linalg.norm(vp3Direction), [10]))
    return vp1, vp2, vp3, pp, roadPlane, focal


def intersection(line_1, line_2) -> Tuple[float, float]:
    det_of_two_lines = np.linalg.det([line_1[:2], line_2[:2]])
    # Check if lines aren't parallel
    if det_of_two_lines != 0:
        # Cramer's rule
        det_x = np.linalg.det([[line_1[2], line_1[1]], [line_2[2], line_2[1]]])
        det_y = np.linalg.det([[line_1[0], line_1[2]], [line_2[0], line_2[2]]])
        x = det_x / det_of_two_lines
        y = det_y / det_of_two_lines
        return np.float32(x), np.float32(y)
    else:
        return False


def line(point_1, point_2):
    """
    Calculating standard line equation from two points Ax + By + C = 0
    :param point_1:
    :param point_2:
    :return: A, B, C points of line equation Ax + By + C = 0
    """
    A = point_1[1] - point_2[1]
    B = point_2[0] - point_1[0]
    C = point_1[0] * point_2[1] - point_2[0] * point_1[1]
    return A, B, -C


def get_transform_matrix(
        vanishing_points, image, im_w, im_h, pts=None, enforce_vp1=True, vp_top=None
):
    if pts is None:
        pts = [
            [0, 0],
            [image.shape[1], 0],
            [image.shape[1], image.shape[0]],
            [0, image.shape[0]],
        ]

    vp1_p1, vp1_p2 = find_corner_pts(vanishing_points[0], pts)
    vp2_p1, vp2_p2 = find_corner_pts(vanishing_points[1], pts)

    # right side
    vp1_l1 = line(vanishing_points[0], pts[vp1_p1])
    # left side
    vp1_l2 = line(vanishing_points[0], pts[vp1_p2])
    # right side
    vp2_l1 = line(vanishing_points[1], pts[vp2_p1])
    # left side
    vp2_l2 = line(vanishing_points[1], pts[vp2_p2])

    # [[top_left], [bottom_left], [bottom_right], [top_right]]
    t_dpts = [[0, 0], [0, im_h], [im_w, im_h], [im_w, 0]]

    intersection_points = [
        intersection(vp1_l1, vp2_l1),
        intersection(vp1_l2, vp2_l1),
        intersection(vp1_l1, vp2_l2),
        intersection(vp1_l2, vp2_l2),
    ]

    if enforce_vp1:

        if vanishing_points[0][1] > im_h:
            t_dpts = [[im_w, 0], [im_w, im_h], [0, im_h], [0, 0]]

        transformed_intersection_points = np.zeros((4, 2), dtype=np.float32)
        t_pts = np.array(t_dpts, np.float32)

        if intersection_points[0][1] < intersection_points[2][1]:
            transformed_intersection_points[0, :] = intersection_points[0]
            transformed_intersection_points[1, :] = intersection_points[2]
        else:
            transformed_intersection_points[0, :] = intersection_points[2]
            transformed_intersection_points[1, :] = intersection_points[0]
        if intersection_points[1][1] < intersection_points[3][1]:
            transformed_intersection_points[3, :] = intersection_points[1]
            transformed_intersection_points[2, :] = intersection_points[3]
        else:
            transformed_intersection_points[3, :] = intersection_points[3]
            transformed_intersection_points[2, :] = intersection_points[1]

        if vp_top is not None:
            transformed_intersection_points = np.roll(
                transformed_intersection_points, -1, axis=0
            )
            for roll in range(4):
                transformed_intersection_points = np.roll(
                    transformed_intersection_points, 1, axis=0
                )
                M = cv2.getPerspectiveTransform(transformed_intersection_points, t_pts)
                vp_top_t = cv2.perspectiveTransform(np.array([[vp_top]]), M)
                if vp_top_t[0, 0, 1] < 0:
                    return cv2.getPerspectiveTransform(
                        transformed_intersection_points, t_pts
                    ), cv2.getPerspectiveTransform(
                        t_pts, transformed_intersection_points
                    )
        else:
            return cv2.getPerspectiveTransform(
                transformed_intersection_points, t_pts
            ), cv2.getPerspectiveTransform(t_pts, transformed_intersection_points)


def get_transform_matrix_with_criterion(
        vp1, vp2, mask, im_w, im_h, constraint=0.8, enforce_vp1=True, vp_top=None
):
    pts = get_points_from_mask(mask, (vp1, vp2))

    image = 255 * np.ones([mask.shape[0], mask.shape[1]])
    M, IM = get_transform_matrix(
        (vp1, vp2), mask, im_w, im_h, pts=pts, enforce_vp1=enforce_vp1, vp_top=vp_top
    )
    t_image = cv2.warpPerspective(mask, M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT)

    while cv2.countNonZero(t_image) / (im_w * im_h) < constraint:
        mask = mask[:-5, :]
        pts = get_points_from_mask(mask, (vp1, vp2))

        # print(pts)
        M, IM = get_transform_matrix(
            (vp1, vp2), mask, im_w, im_h, pts, enforce_vp1=enforce_vp1, vp_top=vp_top
        )
        t_image = cv2.warpPerspective(
            image, M, (im_w, im_h), borderMode=cv2.BORDER_CONSTANT
        )

    return M, IM


def is_right(line_1, line_2, point):
    """
    Check if point is on the right side of the line
    :param line_1:
    :param line_2:
    :param point:
    :return: True if cross product is negative. Means that point is on the right side of the line
    """
    cross_product = np.cross(point - line_1, line_2 - line_1)
    return cross_product < 0


def find_corner_pts(vanishing_point, points):
    """
    Find appropriate corner points
    :param vanishing_point: A point (x, y) representing the vanishing point
    :param points: A list of points [(x1, y1), (x2, y2), ...]
    :return: Two indices P1 and P2, representing the corner points
    """
    # Convert the points list to a NumPy array
    pts = np.array(points)

    # Find P1
    index_corner1 = next(
        idx for idx, pt1 in enumerate(pts)
        if all(np.array_equal(pt2, pt1) or not is_right(vanishing_point, pt1, pt2) for pt2 in pts)
    )

    # Find P2
    index_corner2 = next(
        idx for idx, pt2 in enumerate(pts)
        if all(np.array_equal(pt1, pt2) or is_right(vanishing_point, pt2, pt1) for pt1 in pts)
    )

    return index_corner1, index_corner2


def get_points_from_mask(mask, vanishing_points: tuple):
    """
    Return corner points from road mask
    :param mask: mask of the road
    :param vanishing_points: The two of vanishing points of the image
    :return:
    """
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    contours = np.array(contours[0][:, 0, :])

    # Convex Hull outermost boundary of a set of points in a plane.
    convex_hull = cv2.convexHull(contours)
    pts = convex_hull[:, 0, :]

    idx1, idx2 = find_corner_pts(vanishing_points[0], pts)
    idx3, idx4 = find_corner_pts(vanishing_points[1], pts)
    pts = pts[[idx1, idx2, idx3, idx4]]

    return [pts[0], pts[3], pts[2], pts[1]]


def draw_box(box, fub, image_b, vp0_t, vp1, vp2, vp3, inverse_transform_matrix):
    # TODO refactor to work with batch. Without for loop
    for b, f in zip(box, fub):
        bb_tt, center = get_bb(b, f[0], vp0_t, vp1, vp2, vp3, inverse_transform_matrix)
        bb_tt = [tuple(int(x) for x in point) for point in bb_tt]

        image_b = cv2.line(image_b, bb_tt[0], bb_tt[1], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[1], bb_tt[2], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[2], bb_tt[3], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[3], bb_tt[0], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[0], bb_tt[4], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[1], bb_tt[5], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[2], bb_tt[6], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[3], bb_tt[7], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[4], bb_tt[5], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[5], bb_tt[6], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[6], bb_tt[7], (255, 255, 0), 2)
        image_b = cv2.line(image_b, bb_tt[7], bb_tt[4], (255, 255, 0), 2)

        image_b = cv2.circle(image_b, (int(center[0]), int(center[1])), 5, (0, 255, 255), 5)

    return image_b, center


def get_bb(box, fub, vp0_t, vp1, vp2, vp3, inverse_transform_matrix):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    cy_0 = fub * (ymax - ymin) + ymin
    bb_t = []
    if vp0_t[1] < ymin:
        if (xmin < vp0_t[0]) and (vp0_t[0] < xmax):
            # print("Case 1")
            cy = cy_0
            cx = xmin
            bb_t.append([cx, cy])
            bb_t.append([xmax, cy])
            bb_t.append([xmax, ymax])
            bb_t.append([cx, ymax])
            tx, ty = intersection(line([cx, cy], vp0_t), line([xmin, ymin], [xmin + 1, ymin]))
            bb_t.append([tx, ymin])
            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, inverse_transform_matrix)
            bb_tt = [point[0] for point in bb_tt]

            center = (bb_tt[3] + bb_tt[2]) / 2

            bb_tt.append(intersection(line(bb_tt[1], vp1), line(bb_tt[4], vp2)))
            bb_tt.append(intersection(line(bb_tt[2], vp1), line(bb_tt[5], vp3)))
            bb_tt.append(intersection(line(bb_tt[3], vp1), line(bb_tt[6], vp2)))

        elif vp0_t[0] < xmin:
            # print("Case 2")
            line1 = line([xmin, ymin], vp0_t)
            line2 = line([0, cy_0], [1, cy_0])
            cx, cy = intersection(line1, line2)
            bb_t.append([cx, cy])
            bb_t.append([xmax, cy])
            bb_t.append([xmax, ymax])
            bb_t.append([cx, ymax])
            bb_t.append([xmin, ymin])

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, inverse_transform_matrix)
            bb_tt = [point[0] for point in bb_tt]
            center = (bb_tt[3] + bb_tt[2]) / 2

            bb_tt.append(intersection(line(bb_tt[1], vp1), line(bb_tt[4], vp2)))
            bb_tt.append(intersection(line(bb_tt[2], vp1), line(bb_tt[5], vp3)))
            bb_tt.append(intersection(line(bb_tt[3], vp1), line(bb_tt[6], vp2)))
        else:  # vp0_t[0] > xmax
            # print("Case 3")
            cx, cy = intersection(line([xmax, ymin], vp0_t), line([0, cy_0], [1, cy_0]))
            bb_t.append([cx, cy])
            bb_t.append([cx, ymax])
            bb_t.append([xmin, ymax])
            bb_t.append([xmin, cy])
            bb_t.append([xmax, ymin])

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, inverse_transform_matrix)
            bb_tt = [point[0] for point in bb_tt]
            center = (bb_tt[1] + bb_tt[2]) / 2

            bb_tt.append(intersection(line(bb_tt[1], vp1), line(bb_tt[4], vp3)))
            bb_tt.append(intersection(line(bb_tt[2], vp1), line(bb_tt[5], vp2)))
            bb_tt.append(intersection(line(bb_tt[3], vp1), line(bb_tt[6], vp3)))
    # elif (vp0_t[1] > ymax):
    else:
        if (xmin < vp0_t[0]) and (vp0_t[0] < xmax):
            # print("Case 4")
            cy = cy_0
            bb_t.append([xmin, cy])
            bb_t.append([xmax, cy])
            bb_t.append(intersection(line(vp0_t, [xmax, cy]), line([0, ymax], [1, ymax])))
            bb_t.append(intersection(line(vp0_t, [xmin, cy]), line([0, ymax], [1, ymax])))
            bb_t.append([xmin, ymin])
            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, inverse_transform_matrix)
            bb_tt = [point[0] for point in bb_tt]
            center = (bb_tt[2] + bb_tt[3]) / 2

            bb_tt.append(intersection(line(bb_tt[1], vp3), line(bb_tt[4], vp2)))
            bb_tt.append(intersection(line(bb_tt[2], vp3), line(bb_tt[5], vp1)))
            bb_tt.append(intersection(line(bb_tt[3], vp3), line(bb_tt[4], vp1)))

        elif vp0_t[0] < xmin:
            # print("Case 5")
            cx, cy = intersection(line([xmin, ymax], vp0_t), line([0, cy_0], [1, cy_0]))
            bb_t.append([cx, cy])
            bb_t.append([xmax, cy])
            bb_t.append(list(intersection(line([xmax, cy], vp0_t), line([xmin, ymax], [xmax, ymax]))))
            bb_t.append([xmin, ymax])

            bb_t.append([cx, ymin])
            bb_t.append([xmax, ymin])
            bb_t.append(
                list(intersection(line(bb_t[2], [bb_t[2][0], bb_t[2][1] + 1]), line(vp0_t, [xmax, ymin]))))
            bb_t.append(list(intersection(line(bb_t[6], [bb_t[6][0] + 1, bb_t[6][1]]), line([xmin, 0], [xmin, 1]))))

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, inverse_transform_matrix)
            bb_tt = [point[0] for point in bb_tt]
            center = (bb_tt[3] + bb_tt[2]) / 2
        else:
            # print("Case 6: {}".format(vp0_t))
            print("xmax: {}, xmin: {}, ymin:{}, ymax:{}, c:{}".format(xmax, xmin, ymin, ymax, cy_0))
            cx, cy = intersection(line([xmax, ymax], vp0_t), line([0, cy_0], [1, cy_0]))

            bb_t.append([xmin, cy])
            bb_t.append([cx, cy])
            bb_t.append([xmax, ymax])
            bb_t.append(list(intersection(line([xmin, cy], vp0_t), line([xmin, ymax], [xmax, ymax]))))

            bb_t.append([xmin, ymin])
            bb_t.append([cx, ymin])
            bb_t.append(list(intersection(line([xmax, ymin], [xmax, ymax]), line(vp0_t, bb_t[5]))))
            bb_t.append(list(intersection(line(bb_t[6], [bb_t[6][0] + 1, bb_t[6][1]]), line(vp0_t, bb_t[4]))))

            bb_t_array = np.array([[point] for point in bb_t], np.float32)
            bb_tt = cv2.perspectiveTransform(bb_t_array, inverse_transform_matrix)
            bb_tt = [point[0] for point in bb_tt]
            center = (bb_tt[2] + bb_tt[3]) / 2
    return bb_tt, center


def get_world_coordinates_on_road_plane(p, focal, road_plane, pp):
    p = p / p[2]
    pp = pp / pp[2]

    pp_w = np.append(pp[:2], 0)
    p_w = np.append(p[:2], focal)

    dir_vec = p_w - pp_w
    t = -np.dot(road_plane, np.append(pp_w, 1)) / np.dot(road_plane[:3], dir_vec)

    return pp_w + t * dir_vec


def calculate_iou(box_1, box_2):
    """
    Calculate intersection over union (IoU) between two bounding boxes.

    Args:
        box_1 (tuple): (x1, y1, x2, y2) representing the coordinates of the first bounding box.
        box_2 (tuple): (x1, y1, x2, y2) representing the coordinates of the second bounding box.

    Returns:
        float: The IoU value between the two bounding boxes.
    """
    # Get the coordinates of the intersection rectangle
    x1 = max(box_1[0], box_2[0])
    y1 = max(box_1[1], box_2[1])
    x2 = min(box_1[2], box_2[2])
    y2 = min(box_1[3], box_2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both rectangles
    box_1_area = (box_1[2] - box_1[0] + 1) * (box_1[3] - box_1[1] + 1)
    box_2_area = (box_2[2] - box_2[0] + 1) * (box_2[3] - box_2[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(box_1_area + box_2_area - intersection_area)

    return iou
