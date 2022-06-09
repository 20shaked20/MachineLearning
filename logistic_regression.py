import pandas as pd
import numpy as np


# calculation methods
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


def gradient(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]


def logistic_regression(X, y, theta, alpha, iters):
    cost_array = np.zeros(iters)
    for i in range(iters):
        h = sigmoid(np.dot(X, theta))
        curr_cost = cost(h, y)
        cost_array[i] = curr_cost
        gradient_val = gradient(X, h, y)
        theta = theta - (gradient_val * alpha)
    return theta, cost_array


if __name__ == '__main__':
    data_model = pd.read_csv('../../PycharmProjects/LinearRegression/prices.csv', nrows=24)

    # Mapping to array from csv
    # 85% modelings
    data_x_model = data_model[
        ['LocalPrice', 'Bathrooms', 'SiteSize', 'LivingSize', 'Garages', 'Rooms',
         'Bedrooms', 'Age', 'CtorType',
         'AdrType', 'SalePrice']]
    data_y_model = data_model['Firefighter']

    # add 1 column to allow vectorized calculations ( this is the w )
    data_x_model = np.concatenate((np.ones((data_x_model.shape[0], 1)), data_x_model), axis=1)

    # Generate random w, or with 0 for your choice.
    w_theta = np.zeros(data_x_model.shape[1])

    # define params
    alpha = 0.01
    iterations = 100

    # start value
    h = sigmoid(np.dot(data_x_model, w_theta))
    print("Initial cost value for theta values {0} is: {1}".format(w_theta, cost(h, data_y_model)))

    # run logistic regression
    w_theta, cost_num = logistic_regression(X=data_x_model, y=data_y_model, theta=w_theta, alpha=alpha,
                                            iters=iterations)

    # final values
    h = sigmoid(np.dot(data_x_model, w_theta))
    print("Final cost value for theta values {0} is: {1}".format(w_theta, cost(h, data_y_model)))

    # TESTING:
    # 15% testings
    data_model = pd.read_csv('/Users/Shaked/PycharmProjects/LinearRegression/prices.csv', )
    data_x_test = np.array([data_model.iloc[:, 0][24:28], data_model.iloc[:, 1][24:28], data_model.iloc[:, 2][24:28],
                            data_model.iloc[:, 3][24:28], data_model.iloc[:, 4][24:28], data_model.iloc[:, 5][24:28],
                            data_model.iloc[:, 6][24:28], data_model.iloc[:, 7][24:28], data_model.iloc[:, 8][24:28],
                            data_model.iloc[:, 9][24:28], data_model.iloc[:, 11][24:28]])
    data_y_test = np.array(data_model.iloc[:, 10][24:28])
    for iter in range(0, 4):
        curr_fire_chance = [data_y_test[iter], data_x_test[0][iter], data_x_test[1][iter], data_x_test[2][iter],
                            data_x_test[3][iter],
                            data_x_test[4][iter], data_x_test[5][iter], data_x_test[6][iter], data_x_test[7][iter],
                            data_x_test[8][iter], data_x_test[9][iter], data_x_test[10][iter]]
        h = sigmoid(np.dot(curr_fire_chance, w_theta))
        print(f'Estimated odd for Firefighter in house.no{iter + 24} :', cost(h, data_y_test))


