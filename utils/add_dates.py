from pykml import parser
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import os
import glob
import math

files = glob.glob('./polygon/*.kml')
polygons = []
dates = []
for file in files :
    date = os.path.split(file)[1][2:-4]
    with open(file, 'r') as f :
        doc = parser.parse(f).getroot()
    poly = []
    for e in doc.Document.Folder.Placemark :
        lng, lat, _ = e.Point.coordinates.text.split(',')
        poly.append((float(lat), float(lng)))
    polygons.append(Polygon(poly))
    dates.append(date)

# Truck
with open('./band41/trucks.csv', 'r') as f :
    lines = f.readlines()
print (len(lines))

f = open('./band41/trucks_w_dates.csv', 'w')
f.write('class, latitude, longitude, date, width(m), length(m), score\n')

for line in lines[1:] :
    cls, lat, lng, w, l, score = line.strip().split(',')
    if not cls == 'truck' :
        print (line)
    lat = float(lat)
    lng = float(lng)
    point = Point(lat, lng)
    result = False
    for i, polygon in enumerate(polygons) :
        contain = polygon.contains(point)
        if contain :
            f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(cls, lat, lng, dates[i], w, l, score))
            result = True
    if not result :
        print ("missed")
f.close()

# Car
with open('./band41/cars.csv', 'r') as f :
    lines = f.readlines()
print (len(lines))

f = open('./band41/cars_w_dates.csv', 'w')
f.write('class, latitude, longitude, date, width(m), length(m), score\n')

for line in lines[1:] :
    cls, lat, lng, w, l, score = line.strip().split(',')
    if not cls == 'car' :
        print (line)
    lat = float(lat)
    lng = float(lng)
    point = Point(lat, lng)
    result = False
    for i, polygon in enumerate(polygons) :
        contain = polygon.contains(point)
        if contain :
            f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(cls, lat, lng, dates[i], w, l, score))
            result = True
    if not result :
        print ("missed")
f.close()

# Others
with open('./band41/others.csv', 'r') as f :
    lines = f.readlines()
print (len(lines))

f = open('./band41/others_w_dates.csv', 'w')
f.write('class, latitude, longitude, date, width(m), length(m), score\n')

for line in lines[1:] :
    cls, lat, lng, w, l, score = line.strip().split(',')
    if not cls == 'others' :
        print (line)
    lat = float(lat)
    lng = float(lng)
    point = Point(lat, lng)
    result = False
    for i, polygon in enumerate(polygons) :
        contain = polygon.contains(point)
        if contain :
            f.write('{}, {}, {}, {}, {}, {}, {}\n'.format(cls, lat, lng, dates[i], w, l, score))
            result = True
    if not result :
        print ("missed")
f.close()
