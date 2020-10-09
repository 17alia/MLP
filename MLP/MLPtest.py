import numpy as np
from ReadNormalizedOptdigitsDataset import ReadNormalizedOptdigitsDataset
from MLPtrain import LReLU
X_training_norm, y_training, X_validation_norm, y_validation, X_testing_norm, y_testing = ReadNormalizedOptdigitsDataset('optdigits_train.txt', 'optdigits_valid.txt', 'optdigits_test.txt')
testing_N = X_testing_norm.shape[0]
combined_test_data = np.concatenate((X_testing_norm, y_testing.reshape(testing_N, 1)),axis=1)

def MLPtest(test_data, W, V): # takes in matrix of X and y concatenated, W the weight matrix from inputs to hidden layer, and V the weight matrix from hidden layer to output
    # fit on validation data and find validation error rates
    Y_testing = np.zeros(shape=(testing_N, 10))
    Z_testing = np.zeros(shape=(testing_N, 15))
    for t in range(testing_N):
        for h in range(15):
            Z_testing[t, h] = LReLU(np.dot(W[1:, h], X_testing_norm[t, :]))
        # Good up to here
        for i in range(10):
            numerator = np.exp((np.dot(V[1:, i], Z_testing[t, :])) + V[0, i])
            denominator = 0
            for idx in range(10):
                denominator += np.exp((np.dot(V[1:, idx], Z_testing[t, :])) + V[0, idx])
            Y_testing[t, i] = numerator / denominator
    # calculate validation error rates
    number_wrong_testing = 0
    for t in range(testing_N):
        if np.argmax(Y_testing[t, :]) != y_testing[t]:
            number_wrong_testing += 1
    print("testing set error rate is: " + str(number_wrong_testing / testing_N) + " for number of hidden units: " + str(15))
    return Z_testing

