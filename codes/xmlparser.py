from lxml import etree
import numpy as np
import glob
import os

# paths to xml files
BASE_PATH = "Labelled Images\correct predictions"
file_paths = glob.glob(os.path.sep.join([BASE_PATH, "*.xml"]))
sample = file_paths[0]

# parsing the xml files
coordinates = []
for file in file_paths:
    with open(file) as fobj:
        xml = fobj.read()
    root = etree.fromstring(xml)
    bndbox_coords = []
    for appt in root.getchildren():
        for elem in appt.getchildren():
            count = 0
            bndbox_coord = []
            for coord in elem.getchildren():
                bndbox_coord.append(int(coord.text))
                count = count + 1
                if count == 4:
                    bndbox_coords.append(bndbox_coord)
                    count = 0
                    bndbox_coord = []
    print(bndbox_coords)
    coordinates.append(bndbox_coords)

# saving coords to disk
coordinates = np.array(coordinates)
np.save("bndbox_coords.npy", coordinates)
