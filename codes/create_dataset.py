import numpy as np
import pandas as pd
import cv2

data_desc = pd.read_csv('data.csv')
image_name_list = data_desc['Image']
damage = data_desc['Damage']
images = []
count = 0
for image_name in image_name_list:
    image = cv2.imread('Nordtank Wind Turbine Dataset/17 + 18/' + image_name)
    image = cv2.resize(image, (224, 224), cv2.INTER_AREA)
    image = image / 255.0
    images.append(image)
    count = count + 1
    print('Imported Image %d' % count)

images = np.array(images).astype(np.float16)
print('Saving to disk...')
# np.save('train_images.npy', images)
np.save('train_labels.npy', damage)