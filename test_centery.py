import os

import numpy as np


def get_gt_centery(path: str):
    gt_centery = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            gt_centery.append(float(line.split(' ')[-1]))
    return gt_centery


def get_pred_centery(path: str):
    pred_centery = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            pred_centery.append(float(line.split(' ')[-1]))
    return pred_centery


def print_gt_pred_for_file(gt, pred):
    filtered_gt = []
    filtered_pred = []
    try:
        for i in range(len(gt)):
            print("GT: ", gt[i], " Pred: ", pred[i], " Diff: ", abs(gt[i] - pred[i]))
            filtered_gt.append(gt[i])
            filtered_pred.append(pred[i])
    except IndexError:
        print("GT: ", gt, "there is no prediction for this ground truth")
    return filtered_gt, filtered_pred


def mean_squared_error(y_true, y_pred):
    squared_errors = (y_true - y_pred) ** 2
    mse = np.mean(squared_errors)
    return mse


def test_centery(labels, predictions):
    inference_files = os.listdir(predictions)
    inference_files = [file for file in inference_files if file.endswith('.txt')]
    all_gt_centery = []
    all_pred_centery = []
    for f in inference_files:
        print("File: ", f)
        gt_centery = get_gt_centery(os.path.join(labels, f))
        pred_centery = get_pred_centery(os.path.join(predictions, f))
        gt, pred = print_gt_pred_for_file(gt_centery, pred_centery)
        all_gt_centery.extend(gt)
        all_pred_centery.extend(pred)
    mse = mean_squared_error(np.array(all_gt_centery), np.array(all_pred_centery))
    print("MSE: ", mse)
