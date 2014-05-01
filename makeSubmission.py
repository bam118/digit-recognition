import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

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
X_train, y_train, X_valid, y_valid = splitDataInTrainValidation(X[:10000], y[:10000], 0.4)
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("Training done")
y_pred = knn.predict(X_valid)
print(metrics.classification_report(y_valid, y_pred))
