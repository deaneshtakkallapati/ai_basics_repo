import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os


def kmeans_segmentation(image_path, n_segments):
    image = Image.open(image_path)
    image_array = np.array(image)
    w, h, d = original_shape = tuple(image_array.shape)
    image_2d = np.reshape(image_array, (w * h, d))
    kmeans = KMeans(n_clusters=n_segments, random_state=0)
    labels = kmeans.fit_predict(image_2d)
    segmented_image = np.reshape(labels, (w, h))

    return segmented_image


segmented_images = []
n_segments = 3
images_dir = 'images'
image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
for image in image_files:
    image_path = os.path.join(images_dir, image)
    segmented_image = kmeans_segmentation(image_path, n_segments)
    segmented_images.append(segmented_image)

columns = 3
plt.figure(figsize=(9, 9))
for i, image in enumerate(segmented_images):
    plt.subplot(int(len(segmented_images)/columns + 1), columns, i + 1)
    plt.imshow(image, cmap='BuGn_r'); plt.title(f'Image {i + 1}'); plt.axis('off')
plt.tight_layout()
plt.show()




