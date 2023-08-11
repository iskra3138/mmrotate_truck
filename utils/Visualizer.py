import argparse
import os, glob
from PIL import Image, ImageDraw
from PIL import ImagePath

def parse_args() :
    parser = argparse.ArgumentParser(
            description = "Arguments for saving results images"
            )
    parser.add_argument('--results_path', type=str, help = 'path having txt files of result of each class')
    parser.add_argument('--imgs_path',  type=str, help = 'path having all images predicted')
    parser.add_argument('--save_path', type=str, help = 'path for saving visualized results')
    parser.add_argument('--ext', type=str, default='png', help = 'extension of image file')
    args = parser.parse_args()
    return args

def main() :
    args = parse_args()
    results_path = args.results_path
    imgs_path = args.imgs_path
    save_path = args.save_path
    ext = args.ext
    classnames = ['car', 'truck', 'others']
    colors = ['blue', 'red', 'green']
    ths = [0.8, 0.6, 0.68]

    if not os.path.exists(save_path) :
        os.makedirs(save_path)

    files = glob.glob(os.path.join(imgs_path, '*.{}'.format(ext)))
    preds = {}
    for classname in classnames :
        preds[classname] = {}
        for imgfile in files :
            filename = os.path.split(imgfile)[1][:-4]
            preds[classname][filename] = []

    for i, classname in enumerate(classnames) :
        result_file = os.path.join(results_path, 'Task1_{:s}.txt'.format(classname))
        with open(result_file, 'r') as f :
            lines = f.readlines()
            for line in lines :
                infos = line.strip().split(' ')
                name = infos[0]
                conf = float(infos[1])
                xy = [float(i) for i in infos[2:]]
                if conf > ths[i] :
                    preds[classname][name].append(xy)   

    width = 2
    for imgfile in files :
        img = Image.open(imgfile)
        filename = os.path.split(imgfile)[1][:-4]
        for i, classname in enumerate(classnames) :
            for xy in preds[classname][filename] :
                image = ImagePath.Path(xy).getbbox()
                img1 = ImageDraw.Draw(img)
                img1.polygon(xy, outline =colors[i], width=width)
        img.save(os.path.join(save_path, '{}.jpg'.format(filename)))

if __name__ == '__main__' :
    main()
