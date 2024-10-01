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
        folder = f"{i:03d}"
        folder = os.path.join(imgFolder, folder, leftOrRight)
        if os.path.exists(folder):
            for img_name in sorted(os.listdir(folder)):
                if img_name != "Thumbs.db":  # Exclude Thumbs.db
                    img_path = os.path.join(folder, img_name)
                    img = load_img(
                        img_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    img_array = img_to_array(img)  # Convert image to NumPy array
                    images.append(img_array)
            # Try to load corresponding mask (from L01 to L10)
            for mask_suffix in range(1, 11):  # Assuming masks can be L01 to L10
                mask_name = (
                    "OperatorA_" + f"S1{i:03d}{leftOrRight}{mask_suffix:02d}.tiff"
                )
                mask_path = os.path.join(maskFolder, mask_name)
                if os.path.exists(mask_path):
                    mask = load_img(
                        mask_path,
                        target_size=img_size,
                        color_mode="grayscale" if grayscale else "rgb",
                    )
                    mask_array = img_to_array(mask)
                    masks.append(mask_array)
    return np.array(images), np.array(masks)


baseFolder = "/content/drive/MyDrive/CASIA-Iris-Interval"
maskFolder = "/content/drive/MyDrive/casia4i"

leftOrRight = "L"
train = load_from_folder(10, leftOrRight, baseFolder, maskFolder, (64, 64), True)
test = load_from_folder(5, leftOrRight, baseFolder, maskFolder, (64, 64), True)
val = load_from_folder(2, leftOrRight, baseFolder, maskFolder, (64, 64), True)


# def load_from_folder(imgFolder, maskFolder, img_size=(64, 64), grayscale=True):
#     images = []
#     masks = []
#     for filename in sorted(os.listdir(imgFolder)):
#         path = os.path.join(imgFolder, filename)
#         img = load_img(
#             path, target_size=img_size, color_mode="grayscale" if grayscale else "rgb"
#         )
#         img_array = img_to_array(img)
#         images.append(img_array)
#     for filename in sorted(os.listdir(maskFolder)):
#         path = os.path.join(maskFolder, filename)
#         img = load_img(
#             path, target_size=img_size, color_mode="grayscale" if grayscale else "rgb"
#         )
#         img_array = img_to_array(img)
#         img_array = (img_array > 0).astype(np.float32)
#         masks.append(img_array)
#     return np.array(images), np.array(masks)


# folder = "IrisDataset"

# leftOrRight = "Left"
# # Left Eye

# if leftOrRight == "Left":
#     leftEyeFolder = os.path.join(folder, "Left Eye")

#     trainFolder = os.path.join(leftEyeFolder, "Training Data")
#     imgFolder = os.path.join(trainFolder, "Image")
#     maskFolder = os.path.join(trainFolder, "Mask")
#     train = load_from_folder(imgFolder, maskFolder, (64, 64), True)

#     testFolder = os.path.join(leftEyeFolder, "Testing Data")
#     imgFolder = os.path.join(testFolder, "Image")
#     maskFolder = os.path.join(testFolder, "Mask")
#     test = load_from_folder(imgFolder, maskFolder, (64, 64), True)

#     valFolder = os.path.join(leftEyeFolder, "Validation Data")
#     imgFolder = os.path.join(valFolder, "Image")
#     maskFolder = os.path.join(valFolder, "Mask")
#     val = load_from_folder(imgFolder, maskFolder, (64, 64), True)
# else:
#     # Right Eye

#     rightEyeFolder = os.path.join(folder, "Right Eye")

#     trainFolder = os.path.join(rightEyeFolder, "Training Data")
#     imgFolder = os.path.join(trainFolder, "Image")
#     maskFolder = os.path.join(trainFolder, "Mask")
#     train = load_from_folder(imgFolder, maskFolder, (64, 64), True)

#     testFolder = os.path.join(rightEyeFolder, "Testing Data")
#     imgFolder = os.path.join(testFolder, "Image")
#     maskFolder = os.path.join(testFolder, "Mask")
#     test = load_from_folder(imgFolder, maskFolder, (64, 64), True)

#     valFolder = os.path.join(rightEyeFolder, "Validation Data")
#     imgFolder = os.path.join(valFolder, "Image")
#     maskFolder = os.path.join(valFolder, "Mask")
#     val = load_from_folder(imgFolder, maskFolder, (64, 64), True)
