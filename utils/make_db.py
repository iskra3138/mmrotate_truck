import argparse
import os, glob
import numpy as np
import math
import torch

def parse_args():
    """Parse parameters."""
    parser = argparse.ArgumentParser(
        description='post processing file')
    parser.add_argument('output', help='outout file path')
    args = parser.parse_args()

    return args


def get_lat_lng(point_x, point_y, zoom=20):
    """
        Generates the latitude, longitude based on X,Y tile coordinate
        and zoom level
        Returns:    An latitude, longitude coordinate
    """

    tile_size = 256

    # Use a left shift to get the power of 2
    # i.e. a zoom level of 2 will have 2^2 = 4 tiles
    numTiles = 1 << zoom

    # Find the longitude given x_point
    lng = ((point_x * tile_size )/numTiles - tile_size/2 ) * (360.0/tile_size)

    # Convert the latitude to radians and take the sine
    A = math.exp(((point_y*tile_size/numTiles) - (tile_size / 2))*2 / -(tile_size / (2 * math.pi)))
    lat = math.asin((A-1) / (A+1))*(180.0/math.pi)

    return lat, lng

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

def main() :
    args = parse_args()
    output = int(args.output)
    classnames = ['car', 'truck', 'others']
    ROOT = '/nas2/YJ/git/mmrotate/band41/db_image{:02}/pp'.format(output)

    labels = []
    for cls in classnames :
        pred_file = os.path.join(ROOT, 'Task1_{}.txt'.format(cls))
        with open(pred_file, 'r') as f :
            lines = f.readlines()
            lat_lng = lines[0].split(' ')[0]
            for line in lines :
                pred = line.strip().split(' ')
                label = pred[2:]
                label.append(cls)
                label.append(pred[1])
            
                labels.append(label)

    start_x, start_y = lat_lng.split('_')
    start_x = int(start_x)
    start_y = int(start_y)
    filename = os.path.join(ROOT, '{}.csv'.format(lat_lng))
    with open(filename, 'w') as f :
        f.write('class, latitude, longitude, width(m), length(m), score\n')
        for label in labels :
            coords = torch.Tensor([float(x) for x in label[:8]])
            cx, cy, w, h, theta = poly2obb_oc(coords)[0].numpy()
            w *= 0.10703125
            h *= 0.10703125
            lat, lng = get_lat_lng(start_x+cx/256, start_y+cy/256, 20)
            cls = label[8]
            score = float(label[9])
            if score > 0.3 :
                db = '{}, {}, {}, {:.1f}, {:.1f}, {:.2f}\n'.format(cls, lat, lng, min(w,h), max(w,h), score)
                f.write(db)


if __name__ == '__main__':
    main()
