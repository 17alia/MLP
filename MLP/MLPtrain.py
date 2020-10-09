import numpy as np
import matplotlib.pyplot as plt
from ReadNormalizedOptdigitsDataset import *
eta = 0.01
def LReLU(x):
    if x < 0:
        return 0.01*x
    else:
        return x


def MLPtrain(training_data, val_data, K, H): # function to train MLP. takes in training data, validation data, K=number of output units, H=number of hidden layers
    X_training_norm, y_training, X_validation_norm, y_validation, X_testing_norm, y_testing = ReadNormalizedOptdigitsDataset('optdigits_train.txt', 'optdigits_valid.txt', 'optdigits_test.txt')
    N = X_training_norm.shape[0] # number of samples in training data
    D = X_training_norm.shape[1] # number of features
    W = np.random.uniform(low=-0.01, high=0.01, size=(D+1, H)) # initialize random array of weights from input to hidden layer
    V = np.random.uniform(low=-0.01, high=0.01, size=(H+1, K)) # initialize random array of weights from hidden layer to outputs
    Z = np.zeros(shape=(N, H))
    Y = np.zeros((N, K))
    combined_trn_data = np.concatenate((X_training_norm, y_training.reshape(N, 1)), axis=1)
    num_epochs = 0
    while num_epochs < 100:
        np.random.shuffle(combined_trn_data)
        X = combined_trn_data[:, :-1]
        y = combined_trn_data[:, -1]
        R = np.zeros(shape=(N, K))
        for t in range(N):
            for i in range(K):
                if combined_trn_data[t, -1] == i:
                    R[t, i] = 1
        for t in range(N):
            for h in range(H):
                Z[t, h] = LReLU(np.dot(W[1:, h], X[t, :]))
            # Good up to here
            for i in range(K):
                numerator = np.exp((np.dot(V[1:, i], Z[t, :]))+V[0, i])
                denominator = 0
                for idx in range(K):
                    denominator += np.exp((np.dot(V[1:, idx], Z[t, :]))+V[0, idx])
                Y[t, i] = numerator/denominator
            # Good up to here
            for i in range(K):
                Z_temp = np.insert(Z[t, :], 0, 1, axis=0)
                delta_v_i = eta*(R[t, i] - Y[t, i])*Z_temp
                V[:, i] += delta_v_i
            # Good up to here
            for h in range(H):
                X_temp = np.insert(X[t, :], 0, 1, axis=0)
                if np.dot(W[:, h], X_temp) < 0:
                    delta_w_h = 0.01*eta*np.dot(R[t, :] - Y[t, :], V[h, :])*X_temp
                    W[:, h] += delta_w_h
                else:
                    delta_w_h = 0.01*eta*np.dot(R[t, :] - Y[t, :], V[h, :])*X_temp
                    W[:, h] += delta_w_h
            # Good up to here
        num_epochs +=1
        #print("number of epochs done: " + str(num_epochs))

    # calculate error rates on training set:
    number_wrong_training = 0
    for t in range(N):
        if np.argmax(Y[t, :]) != combined_trn_data[t, -1]:
            number_wrong_training +=1
    print("training set error rate is: " + str(number_wrong_training/N) + " for number of hidden units: " + str(H))
    # fit on validation data and find validation error rates

    validation_N = y_validation.shape[0]
    Y_validation = np.zeros(shape=(validation_N, K))
    Z_validation = np.zeros(shape=(N, H))
    for t in range(validation_N):
        for h in range(H):
            Z_validation[t, h] = LReLU(np.dot(W[1:, h], X_validation_norm[t, :]))
        # Good up to here
        for i in range(K):
            numerator = np.exp((np.dot(V[1:, i], Z_validation[t, :])) + V[0, i])
            denominator = 0
            for idx in range(K):
                denominator += np.exp((np.dot(V[1:, idx], Z_validation[t, :])) + V[0, idx])
            Y_validation[t, i] = numerator / denominator
    # calculate validation error rates
    number_wrong_validation = 0
    for t in range(validation_N):
        if np.argmax(Y_validation[t, :]) != y_validation[t]:
            number_wrong_validation += 1
    print("validation set error rate is: " + str(number_wrong_validation / validation_N) + " for number of hidden units: " + str(H))
    training_error_rate = number_wrong_training/N
    validation_error_rate = number_wrong_validation/validation_N
    return Z, W, V, (training_error_rate, validation_error_rate)


