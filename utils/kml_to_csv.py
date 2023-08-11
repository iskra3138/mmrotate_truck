from bs4 import BeautifulSoup
import csv


def process_coordinate_string(str):
    """
    Take the coordinate string from the KML file, and break it up into [Lat,Lon,Lat,Lon...] for a CSV row
    """
    space_splits = str.split(" ")
    ret = []
    # There was a space in between <coordinates>" "-80.123...... hence the [1:]
    for split in space_splits[1:]:
        comma_split = split.split(',')
        ret.append(comma_split[1])    # lat
        ret.append(comma_split[0])    # lng
    return ret

def main():
    """
    Open the KML. Read the KML. Open a CSV file. Process a coordinate string to be a CSV row.
    """
    with open('tmp.kml', 'r') as f:
        #s = BeautifulSoup(f)
        s = BeautifulSoup(f, 'xml')
        for coords in s.find_all('coordinates'):
            print (process_coordinate_string(coords.string))

        '''
        with open('out.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile)
            for coords in s.find_all('coordinates'):
                writer.writerow(process_coordinate_string(coords.string))
        '''
if __name__ == "__main__":
    main()
