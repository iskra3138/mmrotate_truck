import argparse
import os
from PIL import Image
import numpy as np
import torch
import cv2
from shapely.geometry import Polygon

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='post processing file')
    parser.add_argument('output', help='outout file path')
    args = parser.parse_args()

    return args

def dist_torch(point1, point2):
    """Calculate the distance between two points.
    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).
    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)

def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes

def compare_instersection(cls1, cls2) :
    polygon1 = Polygon([(cls1[x], cls1[x+1]) for x in range(0,8,2)])
    polygon2 = Polygon([(cls2[x], cls2[x+1]) for x in range(0,8,2)])
    iou = 0.0
    w1,h1,w2,h2 = 0,0,0,0
    if polygon1.intersects(polygon2) :
        intersection = polygon1.intersection(polygon2).area
        cls1_overlap = intersection/polygon1.area
        cls2_overlap = intersection/polygon2.area
        iou = intersection /(polygon1.area + polygon2.area - intersection)

        _, _, w1, h1, _ = poly2obb_oc(torch.Tensor(cls1))[0].numpy()
        w1 *= 0.10703125
        h1 *= 0.10703125
        _, _, w2, h2, _ = poly2obb_oc(torch.Tensor(cls2))[0].numpy()
        w2 *= 0.10703125
        h2 *= 0.10703125

    return iou, min(w1, h1), max(w1,h1), min(w2,h2), max(w2,h2)


def main():
    args = parse_args()
    output = int(args.output)
    classnames = ['car', 'truck', 'others']
    detpath = '/nas2/YJ/git/mmrotate/band41/db_image{:02}'.format(output)
    pp_path = '/nas2/YJ/git/mmrotate/band41/db_image{:02}/pp'.format(output)

    if not os.path.exists(pp_path) :
        os.makedirs(pp_path)

    th = [0.80, 0.60, 0.68]
    width_th = [1.5, 2.0, 1.8] ##<- little impact

    preds = {}
    for classname in classnames :
        preds[classname] = []

    for classname in classnames :
        detfile = os.path.join(detpath, 'Task1_{}.txt'.format(classname))   
        with open(detfile, 'r') as f:
            lines = f.readlines()
        
        for line in lines :
            label = line.strip().split(' ')
            filename = label[0]
            score = label[1]
            coords = [float(x) for x in label[2:]]
            obb = poly2obb_oc(torch.Tensor(coords))[0]
            w = min(obb[2:4]).numpy() * 0.10703125
            h = max(obb[2:4]).numpy() * 0.10703125

            idx = classnames.index(classname)
            if float(score) >= th[idx] and min(w,h) > width_th[idx]:
                preds[classname].append([float(x) for x in label[1:]])

    #cnt = 0
    cars = preds['car'].copy()
    trucks = preds['truck'].copy()
    for car in cars :
        for truck in trucks :
            car_coords = car[1:]
            truck_coords = truck[1:]
            iou, w1,h1, w2,h2 = compare_instersection(car_coords, truck_coords)
            if iou > 0.3 :
                if car[0] > truck[0] :
                    try :
                        preds['truck'].remove(truck)
                    except :
                        print ('truck already has beed removed')
                else :
                    try :
                        preds['car'].remove(car)
                    except :
                        print ('car already has beed removed')

    cars = preds['car'].copy()
    others = preds['others'].copy()
    for car in cars :
        for other in others :
            car_coords = car[1:]
            other_coords = other[1:]
            iou, w1,h1, w2,h2 = compare_instersection(car_coords, other_coords)
            if iou > 0.3 :
                #print ('car vs. others', car[0], other[0])
                if car[0] > other[0] :
                    try :
                        preds['others'].remove(other)
                    except :
                        print ('others already has beed removed')
                else :
                    try :
                        preds['car'].remove(car)
                    except :
                        print ('car already has beed removed')

    trucks = preds['truck'].copy()
    others = preds['others'].copy()
    for truck in trucks :
        for other in others :
            truck_coords = truck[1:]
            other_coords = other[1:]

            iou, w1,h1, w2,h2 = compare_instersection(truck_coords, other_coords)

            if iou > 0.3 :
                #cnt+=1
                if h1-h2 > 2.5 :
                    try : #if truck[0] > other[0] :
                        preds['others'].remove(other)
                    except :
                        print ('others already has beed removed')
                else :
                    try :
                        preds['truck'].remove(truck)
                    except :
                        print ('truck already has beed removed')


    for classname in classnames :
        with open(os.path.join(pp_path, 'Task1_{:s}.txt'.format(classname)), 'w') as f:
            for item in preds[classname] :
                label = filename + ' ' + str(item)[1:-1].replace(',','') +'\n'
                f.write(label)


if __name__ == '__main__':
    main()
