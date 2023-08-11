import argparse
import os
from PIL import Image
import glob
import json
import numpy as np

def parse_args() :
    """ Parse Arguments """
    parser = argparse.ArgumentParser ( 
            description = "arguments of post processing" )
    parser.add_argument('--dota_path', help = 'path have "images" and "labelTxt" paths of dota style')
    parser.add_argument('--ext', type=str, default='jpg', help='extension of images [jpg, png, etc]')
    args = parser.parse_args()

    return args


def save(
        filename,
        shapes,
        imagePath,
        imageHeight,

    imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
    ):
    if otherData is None:
        otherData = {}
    if flags is None:
        flags = {}
    data = dict(
        version="4.5.9",
        flags=flags,
        shapes=shapes,
        imagePath=imagePath,
        imageData=imageData,
        imageHeight=imageHeight,
        imageWidth=imageWidth,
    )
    for key, value in otherData.items():
        assert key not in data
        data[key] = value
    try:
        with open(filename, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        #self.filename = filename
    except Exception as e:
        raise LabelFileError(e)

def main ():
    """
    input
        dota_path: path having "images" and "labelTxt" path in dota style
        ext : extension of image file in "images" path
    output
        'json' files converted from 'txt' files in "labelTxt" path will be saved in "images" path
    """
    args = parse_args()
    ROOT = args.dota_path
    IMG_PATH = os.path.join(ROOT, 'images')
    LBL_PATH = os.path.join(ROOT, 'labelTxt')

    ext = args.ext
    labels = glob.glob(os.path.join(LBL_PATH, '*.txt'))
    for label in labels :
        name = os.path.split(label)[1][:-4]
        IMG_NAME = '{}.{}'.format(name, ext)
        img = Image.open(os.path.join(IMG_PATH, IMG_NAME))
        w, h = img.size
        with open(label, 'r') as f :
            lines = f.readlines()
            shapes = []
            for line in lines :
                gt = line.split(' ')
                if len(gt) == 10 :
                    coords = [float(x) for x in gt[:8]]
                    cls = gt[8]
                    data = {}
                    data['label'] = cls
                    data['points'] = [coords[0:2], coords[2:4], coords[4:6], coords[6:8]]
                    data['group_id'] = None
                    data['shape_type'] = "polygon"
                    data["flasgs"] = {}
                    shapes.append(data)

                    save(
                        filename=os.path.join(IMG_PATH, '{}.json'.format(name)),
                        shapes=shapes,
                        imagePath= name + '.{}'.format(ext),
                        imageHeight = h,
                        imageWidth = w,
                        imageData=None,
                        otherData=None,
                        flags=None,
                    )

if __name__ == '__main__' :
    main()
