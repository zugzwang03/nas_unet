import model
import dataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

test = dataLoader.test
best_model = model.best_model


# Predict masks
predictions = best_model.predict(test[0])

# Post-process predictions
# Assuming masks are binary (0 or 1), threshold the predictions
predictions = (predictions > 0.28).astype(np.uint8)


def save_predictions(predictions, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, pred in enumerate(predictions):
        # Convert the prediction to an image
        pred_image = pred.squeeze()  # Remove single-dimensional entries
        pred_image = (pred_image * 255).astype(np.uint8)  # Scale to [0, 255] for saving as image
        output_path = os.path.join(output_folder, f'pred_{i}.png')
        plt.imsave(output_path, pred_image, cmap='gray')  # Save as grayscale image

# Define output folder for predictions
if dataLoader.leftOrRight == 'Left':
    output_folder = 'IrisDataset/Left Eye/Left Eye Output'
else:
    output_folder = 'IrisDataset/Right Eye/Right Eye Output'  # Replace with your desired output folder path
save_predictions(predictions, output_folder)