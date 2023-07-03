import json
from typing import Tuple

import cv2
import numpy as np


def get_calibration_points(calibration_file: str):
    json_file = open(calibration_file, "r")
    calibrations = json.load(json_file)
    json_file.close()
    return (
        calibrations["camera_calibration"]["vp1"],
        calibrations["camera_calibration"]["vp2"],
        calibrations["camera_calibration"]["pp"],
    )


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


def get_focal_length(vp1, vp2, pp):
    return np.sqrt(-np.dot(vp1 - pp, vp2 - pp))


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
    for P1 in range(len(pts)):
        bad = False
        for idx in range(len(pts)):
            if not np.array_equal(pts[idx], pts[P1]) and is_right(
                vanishing_point, pts[P1], pts[idx]
            ):
                bad = True
                break
        if not bad:
            break

    for P2 in range(len(pts)):
        bad = False
        for idx in range(len(pts)):
            if not np.array_equal(pts[idx], pts[P2]) and not is_right(
                vanishing_point, pts[P2], pts[idx]
            ):
                bad = True
                break
        if not bad:
            break

    return P1, P2


def get_points_from_mask(mask, vanishing_points: tuple):
    """
    Return corner points from road mask
    :param mask: mask of the road
    :param vanishing_points: The two of vanishing points of the image
    :return:
    """
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        -2:
    ]
    contours = np.array(contours[0][:, 0, :])

    # Convex Hull outermost boundary of a set of points in a plane.
    convex_hull = cv2.convexHull(contours)
    pts = convex_hull[:, 0, :]

    idx1, idx2 = find_corner_pts(vanishing_points[0], pts)
    idx3, idx4 = find_corner_pts(vanishing_points[1], pts)
    pts = pts[[idx1, idx2, idx3, idx4]]

    return [pts[0], pts[3], pts[2], pts[1]]


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

    # image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # for p in intersection_points:
    #     image = cv2.circle(image, (int(p[0]), int(p[1])), 40, (0, 0, 255), thickness=3)
    # cv2.imshow("Mask with pts", image)
    # cv2.waitKey(0)

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
    # For now we don't care about this. YANGI
    # intersection_points = np.array(intersection_points, np.float32)
    #
    # x_order = np.argsort(intersection_points[:, 0])
    # set_x_left = set(x_order[:2])
    # set_x_right = set(x_order[2:])
    # y_order = np.argsort(intersection_points[:, 1])
    # set_y_top = set(y_order[:2])
    # set_y_bottom = set(y_order[2:])
    #
    # if enforce_vp1:
    #     vp1_dists = [
    #         np.sqrt(
    #             (intersection_points[idx][0] - vp1[0]) ** 2
    #             + (intersection_points[idx][1] - vp1[1]) ** 2
    #         )
    #         for idx in range(4)
    #     ]
    #     y_order = np.argsort(vp1_dists)
    #     set_y_top = set(y_order[:2])
    #     set_y_bottom = set(y_order[2:])
    #
    # res = []
    # res.append(set_x_left.intersection(set_y_top).pop())
    # res.append(set_x_left.intersection(set_y_bottom).pop())
    # res.append(set_x_right.intersection(set_y_bottom).pop())
    # res.append(set_x_right.intersection(set_y_top).pop())
    #
    # t_pts = np.array(t_dpts, np.float32)
    # t_ipts = intersection_points[res, :]
    #
    # return cv2.getPerspectiveTransform(t_ipts, t_pts), cv2.getPerspectiveTransform(
    #     t_pts, t_ipts
    # )


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


def blob_boxer(
    image,
    vpu,
    image_height,
):
    _, image = cv2.threshold((200 * image[0]), 127, 254, cv2.THRESH_BINARY)
    image = image.astype(np.uint8)

    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    cnt = contours[0]

    x_min, y_min, w, h = cv2.boundingRect(cnt)
    x_max = x_min + w
    y_max = y_min + h

    hull = cv2.convexHull(cnt)

    pts = np.array(hull[:, 0, :])
    rt, lt = tangent_point_of_polygon(vpu, pts, image_height)

    # bounding box if on the left side of the vanishing point
    if x_max < vpu[0]:
        cy1 = intersection(line([x_min, y_min], [x_min, y_max]), line(vpu, lt))
        if vpu[1] < 0:
            cx = intersection(line([x_min, y_max], [x_max, y_max]), line(vpu, rt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vpu, [x_max, y_min]))
        else:
            cx = intersection(line([x_min, y_min], [x_max, y_min]), line(vpu, rt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vpu, [x_max, y_max]))
    # bounding box if on the right side of the vanishing point
    elif x_min > vpu[0]:
        cy1 = intersection(line([x_max, y_min], [x_max, y_max]), line(vpu, rt))
        if vpu[1] < 0:
            cx = intersection(line([x_min, y_max], [x_max, y_max]), line(vpu, lt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vpu, [x_min, y_min]))
        else:
            cx = intersection(line([x_min, y_min], [x_max, y_min]), line(vpu, lt))
            cy2 = intersection(line(cx, [cx[0], cx[1] + 1]), line(vpu, [x_min, y_max]))
    else:
        cy1 = intersection(line([x_max, y_min], [x_max, y_max]), line(vpu, rt))
        cy2 = intersection(line([x_min, y_min], [x_min, y_max]), line(vpu, lt))

    if vpu[1] < 0:
        cy = min(cy1[1], cy2[1])
    else:
        cy = max(cy1[1], cy2[1])

    c_c = (cy - y_min) / (y_max - y_min)
    return x_min, x_max, y_min, y_max, c_c


def tangent_point_of_polygon(vpu, pts, image_height):
    left_idx = 0
    right_idx = 0
    p = np.asarray(vpu, dtype=np.float64)
    n = len(pts)
    for i in range(1, n):
        if not is_right(p, pts[left_idx], pts[i]):
            left_idx = i
        if is_right(p, pts[right_idx], pts[i]):
            right_idx = i
    if p[1] > image_height:
        return pts[left_idx], pts[right_idx]
    return pts[left_idx], pts[right_idx]


# class_id center_x center_y bbox_width bbox_height -> yolo format
def convert_to_yolo_annotation_format(
    x_min, x_max, y_min, y_max, image_width, image_height
):
    x_center = ((x_min + x_max) / 2) / image_width
    y_center = ((y_min + y_max) / 2) / image_height
    bbox_width = (x_max - x_min) / image_width
    bbox_height = (y_max - y_min) / image_height
    return (
        x_center,
        y_center,
        bbox_width,
        bbox_height,
    )
