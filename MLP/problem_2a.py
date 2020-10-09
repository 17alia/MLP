from MLPtrain import MLPtrain
from MLPtest import MLPtest
import numpy as np
import matplotlib.pyplot as plt
from ReadNormalizedOptdigitsDataset import ReadNormalizedOptdigitsDataset
X_training_norm, y_training, X_validation_norm, y_validation, X_testing_norm, y_testing = ReadNormalizedOptdigitsDataset('optdigits_train.txt', 'optdigits_valid.txt', 'optdigits_test.txt')
testing_N = X_testing_norm.shape[0]
combined_test_data = np.concatenate((X_testing_norm, y_testing.reshape(testing_N, 1)), axis=1)


Z3, W3, V3, (training_error_rate3, validation_error_rate3) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 3)
Z6, W6, V6, (training_error_rate6, validation_error_rate6) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 6)
Z9, W9, V9, (training_error_rate9, validation_error_rate9) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 9)
Z12, W12, V12, (training_error_rate12, validation_error_rate12) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 12)
Z15, W15, V15, (training_error_rate15, validation_error_rate15) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 15)
Z18, W18, V18, (training_error_rate18, validation_error_rate18) = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, 18)

x_axis = np.array([3, 6, 9, 12, 15, 18])
y_training = np.array([training_error_rate3, training_error_rate6, training_error_rate9, training_error_rate12, training_error_rate15, training_error_rate18])
y_validation = np.array([validation_error_rate3, validation_error_rate6, validation_error_rate9, validation_error_rate12, validation_error_rate15, validation_error_rate18])
plt.scatter(x_axis, y_training) # plots training error rate in blue
plt.scatter(x_axis, y_validation) # plots validation error rate in orange
plt.show()

table_of_error_rates = np.vstack((x_axis, y_training, y_validation))

MLPtest(combined_test_data, W15, V15)