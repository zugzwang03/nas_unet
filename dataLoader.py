import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import matplotlib.pyplot as plt
import numpy as np

def load_from_folder(folder, size, img_size, grayscale):
  dataset=[]
  img_folder = os.path.join(folder, 'Image')
  mask_folder = os.path.join(folder, 'Mask')
  image_filenames = sorted([f for f in os.listdir(img_folder)])
  mask_filenames = sorted([f for f in os.listdir(mask_folder)])
  for i in range(size):
    image_filename = os.path.join(img_folder, image_filenames[i])
    img = load_img(image_filename, target_size=img_size, color_mode='grayscale' if grayscale else 'rgb')
    img_array = img_to_array(img)
    mask_filename = os.path.join(mask_folder, mask_filenames[i]) 
    mask = load_img(mask_filename, target_size=img_size, color_mode='grayscale' if grayscale else 'rgb')
    mask_array = img_to_array(mask)
    dataset.append({'image': (img_array), 'mask': (mask_array)})
  return dataset

# folder='IrisDataset/Left Eye/Training Data'
# train = load_from_folder(folder, (64, 64), True)
# train = (train > 0).astype
# (np.float32)

# print(train)
# for idx in range(10):
#     image = (train[idx]['image']).astype(np.float32)
#     mask = (train[idx]['mask']).astype(np.float32)
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')  # To hide axis
#     plt.show()
#     plt.imshow(mask, cmap='gray')
#     plt.axis('off')  # To hide axis
#     plt.show()