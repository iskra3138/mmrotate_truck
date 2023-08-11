import argparse
import os
import glob
import json

def parse_args():
    """ Parse Arguments"""
    parser = argparse.ArgumentParser(
            description = "Transform LabelMe json file to DOTA style txt file"
            )
    parser.add_argument('--json_path', help = 'path have json files')
    parser.add_argument('--det_path', help = 'path to save txt files')
    args = parser.parse_args()

    return args

def main():
    """
    input
        json_path: path having LabelMe Style of json files
        det_path : path to save txt files
    output
        'txt' files converted from 'json' files will be saved to 'det_path'.
    """
    args = parse_args()
    json_path = args.json_path
    DET = args.det_path

    json_labels = glob.glob(os.path.join(json_path, "*.json"))

    if not os.path.exists(DET) :
        os.makedirs(DET)

    for json_label in json_labels :
        txt_name = os.path.split(json_label)[1].replace('json','txt')

        with open(json_label, 'r') as f :
            anno = json.load(f)

        with open(os.path.join(DET, txt_name), 'w') as f:
            for obj in anno['shapes'] :
                cls = obj['label']
                point = obj['points']

                assert len(point) ==4, "polygon should have 4 points. but this file {} has {} points".format(anno['imagePath'], len(point))
                txt_label = "{:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {} 0\n".format(
                    point[0][0], point[0][1],
                    point[1][0], point[1][1],
                    point[2][0], point[2][1],
                    point[3][0], point[3][1],
                    cls)
                f.write(txt_label)

if __name__ == '__main__' :
    main()
