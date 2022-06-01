import numpy as np

data_x = np.array([[2, 4], [3, 9], [4, 16], [6, 36], [7, 49]])
data_y = np.array([70, 110, 165, 390, 550])
w1 = 0
w2 = 0
b = 0
alpha = 0.001
for iteration in range(10):
    deriv_b = np.mean(1 * ((w1 * data_x[:, 0] + w2 * data_x[:, 1] + b) - data_y))
    # print(deriv_b)
    deriv_w1 = np.dot(((w1 * data_x[:, 0] + w2 * data_x[:, 1] + b) - data_y), data_x[:, 0]) * 1.0 / len(data_y)
    deriv_w2 = np.dot(((w1 * data_x[:, 0] + w2 * data_x[:, 1] + b) - data_y), data_x[:, 1]) * 1.0 / len(data_y)
    b -= alpha * deriv_b
    # print(b)
    w1 -= alpha * deriv_w1
    w2 -= alpha * deriv_w2
print("Estimated price for Galaxy S5: ", np.dot(np.array([5, 25]), np.array([w1, w2])) + b)
print("Estimated price for Galaxy S1: ", np.dot(np.array([1, 1]), np.array([w1, w2])) + b)