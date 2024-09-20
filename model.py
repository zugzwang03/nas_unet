import nas
import main
import numpy as np

best_model = nas.tuner.get_best_models(num_models=1)[0]

for idx in range(10): 
    image_array = main.train[idx]['image']
    mask_array = main.train[idx]['mask']
    image_batch = np.expand_dims(image_array, axis=0)  # Shape: (1, 64, 64, 1) for grayscale
    mask_batch = np.expand_dims(mask_array, axis=0)    # Shape: (1, 64, 64, 1) for segmentation

    # Fit the model with a single image and mask
    history = best_model.fit(image_batch, mask_batch, batch_size=8, epochs=10, validation_split=0.1)
# Train the U-Net model
# history = best_model.fit(np.array(main.train[0]['image']), np.array(main.train[0]['mask']), batch_size=8, epochs=10, validation_split=0.1)

# Evaluate the best model
loss, accuracy = best_model.evaluate(main.val[0]['image'], main.val)  # Assuming you have a validation set
print(f"Best Model - Loss: {loss}, Accuracy: {accuracy}")

# y_out = best_model.predict(main.val)
# print(y_out[0])
# y_out = (y_out > 0.21).astype(np.float32)
# plt.imshow(y_out[0], cmap='gray')
# plt.axis('off')  # To hide axis
# plt.show()

# print(y_out[0])
# # Optionally save the best model
# best_model.save('best_unet_model.h5')