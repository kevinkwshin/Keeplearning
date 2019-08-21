import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
import numpy as np

def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:
    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives
    return: tuple (FP, FN, TP, TN)
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def dice_score(prediction, groundtruth):
    '''
    Dice Similarity Coefficient (F1 Score)
    '''
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    if np.isnan(d):
        return 0.0
    return d

def jaccard_score(prediction, groundtruth):
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0

def hausdorff_score(prediction, groundtruth):
    return spatial.distance.directed_hausdorff(prediction, groundtruth)[0]

def precision_score(prediction, groundtruth):
    # PPV
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision * 100.0

def recall_score(prediction, groundtruth):
    # TPR, sensitivity
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0

def specificity_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0

def intersection_over_union(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0

def accuracy_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = FP + FN + TP + TN
    TNR = np.divide(TP + TN, N)
    return TNR * 100.0

def threshold_predictions(predictions, thr=0.5):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def f1_score(prediction, groundtruth):
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    N = 2*TP + FP + FN
    TNR = np.divide(2*TP, N)
    return TNR * 100.0

# def numeric_score(prediction, groundtruth):
#     FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
#     FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
#     TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
#     TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
#     return FP, FN, TP, TN 
  
# def accuracy(prediction, groundtruth):
#     FP, FN, TP, TN = numeric_score(prediction, groundtruth)
#     N = FP + FN + TP + TN
#     accuracy = np.divide(TP + TN, N)
#     return accuracy * 100.0

def metric_scores_summary(prediction, groundtruth,threshold=0.5):
    """
    threshold = 0 for false
    """
    if threshold == True:
        prediction = threshold_predictions(prediction,threshold)
    
    dice = dice_score(prediction, groundtruth)
    precision = precision_score(prediction, groundtruth)
    recall = recall_score(prediction, groundtruth)
    specificity = specificity_score(prediction, groundtruth)
    iou = intersection_over_union(prediction, groundtruth)
    accuracy = accuracy_score(prediction, groundtruth)
    f1 = f1_score(prediction, groundtruth)
#     hausdorff = hausdorff_score(prediction, groundtruth)
    print("DSC {:.2f} PRECISION {:.2f} RECALL {:.2f} SPECIFICITY {:.2f} IOU {:.2f} ACCURACY {:.2f}".format(dice,precision,recall,specificity,iou,accuracy))
    return dice,precision,recall,specificity,iou,accuracy
