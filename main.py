import model
import dataLoader

test = dataLoader.test
predictions = model.predictions
save_predictions = model.save_predictions


# Define output folder for predictions
if dataLoader.leftOrRight == 'L':
    output_folder = '/content/drive/MyDrive/Fusion/L'
else:
    output_folder = '/content/drive/MyDrive/Fusion/R'  # Replace with your desired output folder path
save_predictions(predictions, output_folder)