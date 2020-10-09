from MLPtrain import MLPtrain
from MLPtest import MLPtest
from MLPtrain import LReLU
from sklearn.decomposition import PCA
Z15, W15, V15, (training_error_rate15, validation_error_rate15) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 15)
import numpy as np
import matplotlib.pyplot as plt
from ReadNormalizedOptdigitsDataset import ReadNormalizedOptdigitsDataset
X_training_norm, y_training, X_validation_norm, y_validation, X_testing_norm, y_testing = ReadNormalizedOptdigitsDataset('optdigits_train.txt', 'optdigits_valid.txt', 'optdigits_test.txt')
X_stacked = np.vstack((X_training_norm, X_validation_norm))
y_stacked = np.hstack((y_training, y_validation))
stacked_N = X_stacked.shape[0]
combined_test_data = np.concatenate((X_stacked, y_stacked.reshape(stacked_N, 1)), axis=1)

# fit on validation data and find validation error rates
Y_stacked = np.zeros(shape=(stacked_N, 10))
Z_stacked = np.zeros(shape=(stacked_N, 15))
for t in range(stacked_N):
    for h in range(15):
        Z_stacked[t, h] = LReLU(np.dot(W15[1:, h], X_stacked[t, :]))
    # Good up to here
    for i in range(10):
        numerator = np.exp((np.dot(V15[1:, i], Z_stacked[t, :])) + V15[0, i])
        denominator = 0
        for idx in range(10):
            denominator += np.exp((np.dot(V15[1:, idx], Z_stacked[t, :])) + V15[0, idx])
        Y_stacked[t, i] = numerator / denominator
# calculate validation error rates
number_wrong_stacked = 0
for t in range(stacked_N):
    if np.argmax(Y_stacked[t, :]) != y_stacked[t]:
        number_wrong_stacked += 1
print("testing set error rate is: " + str(number_wrong_stacked / stacked_N) + " for number of hidden units: " + str(15))
pca2 = PCA(n_components=2)
pca2.fit(X_stacked)
pca3 = PCA(n_components=3)
pca3.fit(X_stacked)

X_transformed_2 = pca2.transform(X_stacked)
X_transformed_3 = pca3.transform(X_stacked)
