import pandas as pd
import numpy as np

def convertDigitsCsvToH5(path):
  digits = pd.read_csv(path, dtype='uint8')
  new_path = path[:-3] + 'h5'
  store = pd.HDFStore(new_path)
  store['digits'] = digits
  store.close()

def loadDigitData(path):
  digits = pd.read_hdf(path, 'digits')
  targets = digits.label.values
  dataset = digits.drop('label', axis=1).values
  return dataset, targets

def splitDataInTrainValidation(dataset, targets, valid_percent):
  number_of_samples = len(dataset)
  number_of_valid_samples = int(number_of_samples * valid_percent)
  np.random.seed(0)
  indices = np.random.permutation(number_of_samples)
  X_train = dataset[indices[:-number_of_valid_samples]]
  y_train = targets[indices[:-number_of_valid_samples]]
  X_valid = dataset[indices[-number_of_valid_samples:]]
  y_valid = targets[indices[-number_of_valid_samples:]]
  return X_train, y_train, X_valid, y_valid
  

#convertDigitsCsvToH5('data/train.csv')
X, y = loadDigitData('data/train.h5')
X_train, y_train, X_valid, y_valid = splitDataInTrainValidation(X, y, 0.2)
print(X_train.shape)
print(y_train.shape)
print(X_valid.shape)
print(y_valid.shape)
