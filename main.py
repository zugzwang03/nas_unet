import test
import dataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

predictions = test.predictions
save_predictions = test.save_predictions


# Define output folder for predictions
if dataLoader.leftOrRight == 'L':
    output_folder = '/content/drive/MyDrive/Iris-Output/Left Eye'
else:
    output_folder = '/content/drive/MyDrive/Iris-Output/Right Eye'  # Replace with your desired output folder path
save_predictions(predictions, output_folder)