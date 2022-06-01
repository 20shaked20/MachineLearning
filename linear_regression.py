import pandas as pd
import numpy as np

data = pd.read_csv('/Users/Shaked/PycharmProjects/LinearRegression/prices.csv'
                   , ',',
                   usecols=['LocalPrice', 'Bathrooms', 'SiteSize', 'LivingSize', 'Garages', 'Rooms', 'Bedrooms', 'Age',
                            'CtorType', 'AdrType', 'Firefighter'])
# data = pd.read_csv()
# print(data.head())
df = data[['LocalPrice', 'Bathrooms', 'SiteSize']]
# print(A)

# Add a column of ones for the bias term.
# I chose 1 because if you multiply one with any value, that value does not change.
df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
# print(df.head())


data_x_with_names = df.drop('LocalPrice', axis='columns')  # ignores the output variable only.

# Mapping to array from csv
data_x = np.array([data_x_with_names['Bathrooms'], data_x_with_names['SiteSize']])
data_y = np.array(df.iloc[:, 1])
w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b = 0
alpha = 0.001


def deriv_w(which_w: int, w_method: list, data_x_method, data_y_method, b_method) -> float:
    sigma = 0
    for k in range(0, 11):
        sigma += w_method[k] * data_x_method[k]
    ans = np.dot(((sigma + b_method) - data_y_method), data_x_method[which_w]) * 1.0 / len(data_y_method)
    return ans


def deriv_b(w_method: list, data_x_method, data_y_method, b_method):
    sigma = 0
    for k in range(0, 11):
        sigma += w_method[k] * data_x_method[k]
    ans = np.mean(1 * ((sigma + b_method) - data_y_method))
    print(ans)
    return ans


# Data iterations:
# for iteration in range(10):
# b -= alpha * deriv_b(w_method=w, data_x_method=data_x, data_y_method=data_y, b_method=b)
# for i in range(0, 11):
#     w[i] -= alpha * deriv_w(which_w=i, w_method=w, data_x_method=data_x, data_y_method=data_y, b_method=b)

print("Estimated price for Galaxy S5: ", np.dot(np.array([5, 25]), np.array([w[0], w[1]])) + b)
print("Estimated price for Galaxy S1: ", np.dot(np.array([1, 1]), np.array([w[0], w[1]])) + b)
