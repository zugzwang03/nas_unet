import model
import dataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

test = dataLoader.test
predictions = model.predictions
save_predictions = model.save_predictions


# Define output folder for predictions
if dataLoader.leftOrRight == 'Left':
    output_folder = 'IrisDataset/Left Eye/Left Eye Output'
else:
    output_folder = 'IrisDataset/Right Eye/Right Eye Output'  # Replace with your desired output folder path
save_predictions(predictions, output_folder)