import nas
import main
import numpy as np
import matplotlib.pyplot as plt
import dataLoader

tuner = nas.tuner
train = dataLoader.train
val = dataLoader.val

best_model = tuner.get_best_models(num_models=1)[0]

# Train the U-Net model
history = best_model.fit(train[0], train[1], batch_size=8, epochs=20, validation_split=0.1)

# Evaluate the best model
loss, accuracy = best_model.evaluate(val[0], val[1])  # Assuming you have a validation set
print(f"Best Model - Loss: {loss}, Accuracy: {accuracy}")

y_out = best_model.predict(val[0])
print(y_out[0])
y_out = (y_out > 0.0021).astype(np.float32)
plt.imshow(y_out[0], cmap='gray')
plt.axis('off')  # To hide axis
plt.show()

print(y_out[0])
# Optionally save the best model
best_model.save('best_unet_model.h5')