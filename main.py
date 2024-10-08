import model
import dataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

test = dataLoader.test
predictions = model.predictions
save_predictions = model.save_predictions


# Define output folder for predictions
if dataLoader.leftOrRight == 'L':
    output_folder = '/content/drive/MyDrive/Iris-Output/Left Eye'
else:
    output_folder = '/content/drive/MyDrive/Iris-Output/Right Eye'  # Replace with your desired output folder path
save_predictions(predictions, output_folder)