import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img  # type: ignore
import matplotlib.pyplot as plt
import numpy as np


def load_from_folder(
    quantity, leftOrRight, imgFolder, maskFolder, img_size=(64, 64), grayscale=True
):
    images = []
    masks = []
    for i in range(1, quantity):
        print(i)
        folder = f"{i:03d}"
        folder = os.path.join(imgFolder, folder, leftOrRight)
        print(folder)
        count = 0
        if os.path.exists(folder):
            for img_name in sorted(os.listdir(folder)):
                if img_name != "Thumbs.db":  # Exclude Thumbs.db
                    img_path = os.path.join(folder, img_name)
                    print(img_path)
                    img = load_img(
                        img_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    count += 1
                    img_array = img_to_array(img)  # Convert image to NumPy array
                    images.append(img_array)
            # Try to load corresponding mask (from L01 to L10)
            for mask_suffix in range(1, count + 1):  # Assuming masks can be L01 to L10
                mask_name = (
                    "OperatorA_" + f"S1{i:03d}{leftOrRight}{mask_suffix:02d}.tiff"
                )
                mask_path = os.path.join(maskFolder, mask_name)
                if os.path.exists(mask_path):
                    print(mask_path)
                    mask = load_img(
                        mask_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    mask_array = img_to_array(mask)
                    mask_array = (mask_array > 0).astype(np.float32)
                    masks.append(mask_array)
    return np.array(images), np.array(masks)


baseFolder = "/content/drive/MyDrive/CASIA-Iris-Interval"
maskFolder = "/content/drive/MyDrive/casia4i"

leftOrRight = "L"
train = load_from_folder(10, leftOrRight, baseFolder, maskFolder, (64, 64), True)
test = load_from_folder(5, leftOrRight, baseFolder, maskFolder, (64, 64), True)
val = load_from_folder(2, leftOrRight, baseFolder, maskFolder, (64, 64), True)
