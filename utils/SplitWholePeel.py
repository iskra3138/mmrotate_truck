import os, glob, shutil
from PIL import Image

ROOT = '/path/to/have/all/tiles'
original_files = glob.glob(os.path.join(ROOT, '*.png'))

longs = []
for file in original_files :
    filename = os.path.split(file)[1]
    _, latitude, longitude = filename[:-4].split('_')
    longs.append(longitude)

longs = list(set(longs))
longs.sort()
start_long = longs[0]
end_long = longs[-1]

file_splits={}
cnt = 0
for i in range(start_long, end_long, 50) : # will be made up of 41 bands
    file_splits[str(i)] = []

for file in original_files :
    filename = os.path.split(file)[1]
    _, latitude, longitude = filename[:-4].split('_')
    idx = ((int(longitude) - start_long) // 50) * 50 + start_long
    file_splits[str(idx)].append(filename)

base_size = 2560
for i, start_y in enumerate(range(start_long, end_long 50)) :
    lats = []
    path = 'db_image{:02}'.format(i)
    final_path = os.path.join(ROOT, 'Peel_Bands_41', path)
    if not os.path.exists(final_path) :
        os.makedirs(final_path)
    
    for file in file_splits[str(start_y)] :
        filename = os.path.split(file)[1]
        _, latitude, _ = filename[:-4].split('_')
        lats.append(latitude)
    start_x = int(min(lats))
    width = int((int(max(lats))-int(min(lats)))/10)
    img =  Image.new(size = (width*base_size, 5*base_size), mode = 'RGB', color = 'white')
    for file in file_splits[str(start_y)] :
        tile_img =  Image.open(os.path.join(ROOT, 'Peel_Tiles_2560', file))
        filename = os.path.split(file)[1]
        _, latitude, longitude = filename[:-4].split('_')
        loc_x = int((int(latitude) - start_x)/10)*base_size
        loc_y = int((int(longitude) - start_y)/10)*base_size
        img.paste(tile_img, (loc_x, loc_y))
    
    img_name = os.path.join(final_path, '{}_{}.png'.format(start_x, start_y))
    img.save(img_name)
    print (i, start_x, start_y)
    
