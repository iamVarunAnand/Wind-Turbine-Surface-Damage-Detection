# import the necessary packages
from imutils import paths
import numpy as np
import cv2
import os

class BoundingBox:
    @staticmethod
    def draw(image, bndbox_coords):
        for bndbox in bndbox_coords:
            print("[INFO] adding bndbox: ", bndbox)
            x1 = bndbox[0]
            y1 = bndbox[1]
            x2 = bndbox[2]
            y2 = bndbox[3]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 10)

        return image
# define path to image files and output path
IMAGE_PATH = "Labelled Images\cp"
image_paths = list(paths.list_images(IMAGE_PATH))
OUTPUT_PATH = "Labelled Images"

# load bndbox coordinates
bndbox_coords = np.load("bndbox_coords.npy", allow_pickle = True)

for (i, image_path) in enumerate(image_paths):
    image = cv2.imread(image_path)
    image = BoundingBox.draw(image, bndbox_coords[i])
    cv2.imwrite(os.path.sep.join([OUTPUT_PATH, image_path.split(os.path.sep)[-1]]), image)
    
    # print updates to screen
    print("[INFO] annotated {}/{}".format(i + 1, len(image_paths)))
