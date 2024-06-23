import numpy as np

def  calc_dice(pred,gt):
    b = pred.shape[0]
    assert pred.max()<=1 and gt.max()<=1
    assert type(pred)==type(gt)
    pred=np.reshape(pred,newshape=(b,-1))
    gt=np.reshape(gt,newshape=(b,-1))
    smooth = 1
    intersection = (pred*gt).sum(axis=1)
    return (2. * intersection + smooth) / (pred.sum(axis=1) + gt.sum(axis=1) + smooth)