import dataLoader

train_folder = 'IrisDataset/Left Eye/Training Data'
test_folder = 'IrisDataset/Left Eye/Testing Data'
validation_folder = 'IrisDataset/Left Eye/Validation Data'

train = dataLoader.load_from_folder(train_folder, 10, (64, 64), True)
test = dataLoader.load_from_folder(test_folder, 5, (64, 64), True)
val = dataLoader.load_from_folder(validation_folder, 2, (64, 64), True)

