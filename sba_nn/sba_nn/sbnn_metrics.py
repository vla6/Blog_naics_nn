###########################################################
##
## Functions related to threshold tuning,
## performance metrics, or other model fit information.
##
############################################################

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, \
    average_precision_score, roc_auc_score

#
# Threshold tuning
#

def get_f1_frame(actual, pred_prob, num_pts = 51):
    """Given actual responses and model probability predictions, get f1 scores over a
    range of thresholds
      Inputs:
        actual: Pandas series containing actual target values (0/1)
        pred_prob:  Pandas series containing probability predictions
        num_pts:  Number of threshold points to examine
      Value: Dataframe consistig of a threshold and f1 score"""
    thresh_ser = pd.Series(np.linspace(0, 1, num_pts))
    return pd.DataFrame({'thresh': thresh_ser,
                         'f1': thresh_ser.apply(lambda x: f1_score(actual, 
                                                                   get_binary_predictions(pred_prob, x)))})

def get_binary_predictions(pred_prob, thresh):
    """Given probability predictions and a decision thresold, return binary predictions
      Inputs:
        pred_prob: Pandas series containing model probability predictions
        thresh:  Threshold above which we predict a 1 outcome
      Value: Series containing binary predictions
    """
    return pd.Series(np.where(pred_prob > thresh, 1, 0),
                    index=pred_prob.index).astype(int)

#
# Standard metrics
#

def do_metric(metric, actual, predict_bin, predict_prob):
    """ Get a sklearn metric, selecting the proper inputs and returning
    np.nan on error
    Inputs:
        metric: sklearn metric. Expected to either be average_precision_score or
          roc_auc_score, or to take the actual and binary predictions as inputs.
        actual: Pandas series containing actual outcomes
        predict_bin: Pandas series containing binary predictions
        predict_prob: Pandas series containing probability predictions
      Value: Metric for the data (float)
    """
    try:
        if metric in [average_precision_score, roc_auc_score]:
            return metric(actual, predict_prob)
        else:
            return metric(actual, predict_bin)
    except:
        return np.nan
    

def dset_metrics(actual, predict_bin = None, predict_prob = None,
                 metrics_list = [accuracy_score, f1_score, precision_score, recall_score, 
                                average_precision_score, roc_auc_score]):
    """ Return a Series containing standard metrics for the binary classification model.
    Inputs:
        actual: Pandas series containing actual outcomes
        predict_bin: Pandas series containing binary predictions.  Optional (but either
          predict_bin or predict_prob will be needed to generate metrics)
        predict_prob: Pandas series containing probability predictions.  Optional (but 
          either predict_bin or predict_prob will be needed to generate metrics)
        metrics_list: List of classification metrics to return
      Value: Series containing metrics data for the inputs 
    """
    return pd.Series([do_metric(m, actual, predict_bin, predict_prob) \
                      for m in metrics_list], index=[m.__name__ for m in metrics_list]) 
