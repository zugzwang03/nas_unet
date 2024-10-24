import nas
import os
import numpy as np
import matplotlib.pyplot as plt
import dataLoader
import tensorflow as tf

tuner = nas.tuner
train = dataLoader.train
val = dataLoader.val
test = dataLoader.test

best_model = tuner.get_best_models(num_models=1)[0]

# Train the U-Net model
history = best_model.fit(train[0], train[1], batch_size=8, epochs=20, validation_split=0.1)

# Evaluate the best model
loss, accuracy = best_model.evaluate(val[0], val[1])  # Assuming you have a validation set
print(f"Best Model - Loss: {loss}, Accuracy: {accuracy}")

y_out = best_model.predict(val[0])

# Optionally save the best model
best_model.save('best_unet_model.h5')

# # Predict masks
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
