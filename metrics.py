import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
import numpy as np

def numeric_score(groundtruth,prediction):
    """Computation of statistical numerical scores:
    * FP = False Positives
    * FN = False Negatives
    * TP = True Positives
    * TN = True Negatives
    return: tuple (FP, FN, TP, TN)
    
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    """
    FP = np.float(np.sum((prediction == 1) & (groundtruth == 0)))
    FN = np.float(np.sum((prediction == 0) & (groundtruth == 1)))
    TP = np.float(np.sum((prediction == 1) & (groundtruth == 1)))
    TN = np.float(np.sum((prediction == 0) & (groundtruth == 0)))
    return FP, FN, TP, TN

def score_dice(groundtruth,prediction):
    '''
    Dice Similarity Coefficient (=F1 Score)

    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    d = (1 - spatial.distance.dice(pflat, gflat)) * 100.0
    if np.isnan(d):
        return 0.0
    return d

def score_jaccard(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    pflat = prediction.flatten()
    gflat = groundtruth.flatten()
    return (1 - spatial.distance.jaccard(pflat, gflat)) * 100.0

def score_hausdorff(groundtruth,prediction):
    return spatial.distance.directed_hausdorff(groundtruth,prediction)[0]

def score_precision(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    FP, FN, TP, TN = numeric_score(groundtruth,prediction)
    if (TP + FP) <= 0.0:
        return 0.0

    precision = np.divide(TP, TP + FP)
    return precision * 100.0

def score_recall(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    FP, FN, TP, TN = numeric_score(groundtruth,prediction)
    if (TP + FN) <= 0.0:
        return 0.0
    TPR = np.divide(TP, TP + FN)
    return TPR * 100.0

def score_specificity(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    FP, FN, TP, TN = numeric_score(groundtruth,prediction)
    if (TN + FP) <= 0.0:
        return 0.0
    TNR = np.divide(TN, TN + FP)
    return TNR * 100.0

def score_iou(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    FP, FN, TP, TN = numeric_score(groundtruth,prediction)
    if (TP + FP + FN) <= 0.0:
        return 0.0
    return TP / (TP + FP + FN) * 100.0

def score_accuracy(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    FP, FN, TP, TN = numeric_score(groundtruth,prediction)
    N = FP + FN + TP + TN
    TNR = np.divide(TP + TN, N)
    return TNR * 100.0

def score_f1(groundtruth,prediction):
    '''
    Reference
    https://en.wikipedia.org/wiki/Receiver_operating_characteristic
    '''
    FP, FN, TP, TN = numeric_score(groundtruth,prediction)
    N = 2*TP + FP + FN
    TNR = np.divide(2*TP, N)
    return TNR * 100.0

def threshold_predictions(predictions, threshold=0.5):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < threshold
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= threshold
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def metric_scores_summary(groundtruth,prediction,threshold=False,print_score=False):
    """
    !!! All values should be onehot endcoded !!!
    - threshold : if no need to get threshold, threshold = False
    - print_score : option for printing metrics
    """
    if threshold != False:
        prediction = threshold_predictions(prediction,threshold)
    
    dice = score_dice(groundtruth,prediction)
    precision = score_precision(groundtruth,prediction)
    recall = score_recall(groundtruth,prediction)
    specificity = score_specificity(groundtruth,prediction)
    iou = score_iou(groundtruth,prediction)
    accuracy = score_accuracy(groundtruth,prediction)
#     hausdorff = hausdorff_score(prediction, groundtruth)  # only work for 2D

    if print_score==True:
        print("DSC {:.2f} PRECISION {:.2f} RECALL {:.2f} SPECIFICITY {:.2f} IOU {:.2f} ACCURACY {:.2f}".format(dice,precision,recall,specificity,iou,accuracy))
    
    return dice,precision,recall,specificity,iou,accuracy
