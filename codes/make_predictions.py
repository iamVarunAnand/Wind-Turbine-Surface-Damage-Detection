# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse
import cv2
import os

# parse the command line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required = True, help = "path to store labelled images")
args = vars(ap.parse_args())

# define the path to the model, data and list image paths
MODEL_PATH = "Models\RS50w32_1.h5"
DATA_PATH  = "Data (npy)"
IMAGES_PATH =  "D:\\ML\\Datasets\\Nordtank Wind Turbine Dataset\\17 + 18"

# import the train data
print("[INFO] loading data from disk...")
data = np.load(os.path.sep.join([DATA_PATH, "train_images.npy"]))
labels = np.load(os.path.sep.join([DATA_PATH, "train_labels.npy"]))
image_paths = list(paths.list_images(IMAGES_PATH))

# split the data and image_paths into train and test sets
(x_train, x_test, y_train, y_test) = train_test_split(data, labels,    test_size = 0.15, random_state = 100)
(train_image_paths, test_image_paths) = train_test_split(image_paths,
                    test_size = 0.15, random_state = 100)

# load the model and compile it
print("[INFO] compiling model...")
model = load_model(MODEL_PATH)
model.compile(loss = "binary_crossentropy", optimizer = "adam")

# evaluate the model
print("[INFO] evaluating model...")
predictions = model.predict(x_test, batch_size = 8)
predictions[predictions < 0.5] = 0
predictions[predictions > 0] = 1
predictions = predictions.astype("int")
print(classification_report(y_test, predictions))

# get the indices of images marked as positive / negative damage
y_test = y_test.reshape((y_test.shape[0], 1))
# indices = np.where(predictions == y_test)
indices = np.where(predictions != y_test)

# save the images to disk
print("[INFO] saving images to disk...")
count = 1
for i in indices[0]:
    image = cv2.imread(test_image_paths[i])
    cv2.putText(image, "Damage [0/1]: {}".format(predictions[i]),
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 15)
    cv2.imwrite(os.path.sep.join([args["output"], str(count) + ".jpg"]), image)

    # print updates to the screen
    print("[INFO] labelled and saved {}/{}".format(count, len(indices[0])))
    count = count + 1
