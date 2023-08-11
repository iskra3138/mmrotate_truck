import argparse
import xml.etree.ElementTree as ET
import os
import glob
import numpy as np
from dotadevkit.polyiou import polyiou


def parse_args() :
    parser = argparse.ArgumentParser(
            description = "Arguments for calculating AP",
            )
    parser.add_argument('--results_path', help='path having txt files of result of each class')
    parser.add_argument('--annos_path', help='path having each annotation files')
    args = parser.parse_args()
    return args
    


## this function is based on the dotadevkit 
## https://github.com/ashnair1/dotadevkit/blob/master/dotadevkit/evaluate/task1.py
def parse_gt(filename):
    """

    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

## this function is given from the dotadevkit 
## https://github.com/ashnair1/dotadevkit/blob/master/dotadevkit/evaluate/task1.py
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


## this function is based on the dotadevkit
## https://github.com/ashnair1/dotadevkit/blob/master/dotadevkit/evaluate/task1.py
def voc_eval(detpath,
             annos,
             classname,
             ovthresh=0.5,
             use_07_metric=False):
    """
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annos: annotations of dota format.
    classname: Category name
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    #with open(imagesetfile, 'r') as f:
    #    lines = f.readlines()
    #imagenames = [x.strip() for x in lines]

    recs = {}
    imagenames = []
    for anno in annos:
        imagename = os.path.split(anno)[1][:-4]
        imagenames.append(imagename)
        recs[imagename] = parse_gt(anno) # GT

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool_)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfilename = 'Task1_{:s}.txt'.format(classname)
    detfile = os.path.join(detpath, detfilename)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.
    # compute precision recall

    #print('check fp:', fp)
    #print('check tp', tp)

    print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
    print ("AP : {:.3f}, PREC: {:.3f}, REC: {:.3f}".format(ap, prec[-1], rec[-1]))
    return sorted_scores, rec, prec, ap, npos, tp, fp


## this function is based on the dotadevkit
## https://github.com/ashnair1/dotadevkit/blob/master/dotadevkit/evaluate/task1.py
def main() :
    args = parse_args()
    results_path = args.results_path
    annos = glob.glob(os.path.join(args.annos_path, '*.txt'))

    classnames = ['car', 'truck', 'others']
    classaps = []
    map = 0
    for classname in classnames:
        print('classname:', classname)
        sorted_scores, rec, prec, ap,  npos, tp, fp = voc_eval(results_path,
             annos,
             classname,
             ovthresh=0.5,
             use_07_metric=False)
        map = map + ap
        #print('ap: ', ap)
        classaps.append(ap)

        max_f = 0
        idx = 0
        for i in range(rec.shape[0]) :
            f_score = 2*(prec[i]*rec[i])/(prec[i]+rec[i])
            if f_score > max_f :
                max_f = f_score
                idx = i
        print ("th:{}, num_preds:{}, total_gt:{}, max_f:{:.3f}, prec:{:.3f}, rec:{:.3f}, num_tp:{}, num_fp:{}".format(
            -sorted_scores[idx], idx+1, npos, max_f, prec[idx], rec[idx], tp[idx], fp[idx]))
       # plt.show()
    map = map/len(classnames)
    print('map:', map)
    classaps = 100*np.array(classaps)
    print('classaps: ', classaps)


if __name__ == '__main__' :
    main()
