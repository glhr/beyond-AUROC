import numpy as np

from sklearn.metrics import roc_auc_score, roc_curve, auc, precision_recall_curve, det_curve


from scipy import interpolate



def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    from sklearn.utils.extmath import stable_cumsum
    from sklearn.exceptions import UndefinedMetricWarning
    import warnings
    """
    modified from sklearn.metrics._binary_clf_curve
    """

    # make y_true a boolean vector
    y_true = y_true == pos_label
    y_neg = ~y_true

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    y_neg = y_neg[desc_score_indices]
    weight = 1.0

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    indices = np.arange(len(y_score))
    threshold_idxs = np.r_[indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true * weight)[threshold_idxs]
    fps = stable_cumsum(y_neg * weight)[threshold_idxs]
    
    thresholds = y_score[threshold_idxs]
    
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]
    thresholds = np.r_[1, thresholds]

    if fps[-1] <= 0:
        warnings.warn(
            "No negative samples in y_true, false positive value should be meaningless",
            UndefinedMetricWarning,
        )
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn(
            "No positive samples in y_true, true positive value should be meaningless",
            UndefinedMetricWarning,
        )
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]
    return fpr, tpr, thresholds

def fnr_at_tnr(preds,labels,pos_label=1,tnr=0.95):
    fprs, tprs, thresholds = roc_curve(labels, preds, pos_label=pos_label)
    tnrs = 1-fprs
    fnrs = 1-tprs
    if all(tnrs < tnr):
        return 0, None
    elif all(tnrs >= tnr):
        idxs = [i for i, x in enumerate(tnrs) if x >= tnr]
        selected_idx = np.argmin(fnrs[idxs])
        return fnrs[idxs][selected_idx], thresholds[idxs][selected_idx]

    thresh_intrp= interpolate.interp1d(tnrs,thresholds)
    thresh = thresh_intrp(tnr)

    fnr_interp = interpolate.interp1d(thresholds,fnrs)
    fnr95 = fnr_interp(thresh)

    # plt.plot(thresholds, tnr, label='TNR')
    # plt.plot(thresholds, fnr, label='FNR')
    # plt.axvline(x = thresh, color = 'black', label = '@TNR95')
    # plt.legend()
    # plt.show()

    return fnr95.item(), thresh.item()

def fpr_at_tpr(preds, labels, pos_label=1, tpr=0.95):
    fprs, tprs, thresholds = roc_curve(labels, preds, pos_label=pos_label)
    if all(tprs < tpr):
        # No threshold allows TPR >= 0.95
        return 0, None
    elif all(tprs >= tpr):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tprs) if x >= tpr]
        selected_idx = np.argmin(fprs[idxs])
        return fprs[idxs][selected_idx], thresholds[idxs][selected_idx]

    thresh_intrp= interpolate.interp1d(tprs,thresholds)
    thresh = thresh_intrp(tpr)

    fpr_interp = interpolate.interp1d(thresholds,fprs)
    fpr = fpr_interp(thresh)

    # plt.plot(thresholds, tnr, label='TNR')
    # plt.plot(thresholds, fnr, label='FNR')
    # plt.axvline(x = thresh, color = 'black', label = '@TNR95')
    # plt.legend()
    # plt.show()

    return fpr.item(), thresh.item()

def calc_standard_metrics(preds,labels,pos_label=1):
    metrics = dict()
    metrics["auroc"] = roc_auc_score(labels,preds)
    metrics["fpr@95tpr"], thresh_95tpr = fpr_at_tpr(preds,labels,pos_label=pos_label,tpr=0.95)
    metrics["fnr@95tnr"], thresh_95tnr = fnr_at_tnr(preds,labels,pos_label=pos_label,tnr=0.95)

    metrics["thresh_95tpr"] = thresh_95tpr
    metrics["thresh_95tnr"] = thresh_95tnr

    precision, recall, thresholds = precision_recall_curve(1-labels,-preds,pos_label=1)
    metrics["aupr-in"] = auc(recall, precision)
    precision, recall, thresholds = precision_recall_curve(labels,preds,pos_label=1)
    metrics["aupr-out"] = auc(recall, precision)
    return metrics

def calc_autc(preds,labels):
    c = dict()

    fpr,tpr,thresholds = _binary_clf_curve(labels,preds,pos_label=1)
    
    fnr = 1-tpr
    
    sorted_idx = thresholds.argsort()
    c["sorted_thresh"]= thresholds[sorted_idx]
    c["sorted_fpr"] = fpr[sorted_idx]
    c["sorted_fnr"] = fnr[sorted_idx]
    
    if not np.isin(0,c["sorted_thresh"]):
        #print("adding 0")
        c["sorted_thresh"] = np.insert(c["sorted_thresh"],0,0)
        c["sorted_fpr"] = np.insert(c["sorted_fpr"],0,1)
        c["sorted_fnr"] = np.insert(c["sorted_fnr"],0,0)
        
    c["aufnr"] = auc(c["sorted_thresh"], c["sorted_fnr"])
    c["aufpr"] = auc(c["sorted_thresh"], c["sorted_fpr"])
    c["autc"] = (c["aufnr"]+c["aufpr"])/2

    return c

    
