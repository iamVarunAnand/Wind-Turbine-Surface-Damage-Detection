import numpy as np
import matplotlib.pyplot as plt

images = np.load('train_images_17.npy')
labels = np.load('train_labels_17.npy')

np.random.seed(5)
indices = np.random.randint(0, images.shape[0], size = (16, 1))

count = 1
for i in indices:
    plt.subplot(4, 4, count).set_title(labels[i])
    plt.imshow(np.squeeze(images[i]).astype(np.float64))
    count = count + 1

plt.show()
