import pandas as pd
import numpy as np

data = pd.read_csv('/Users/Shaked/PycharmProjects/LinearRegression/prices.csv'
                   , ',',
                   usecols=['LocalPrice', 'Bathrooms', 'SiteSize', 'LivingSize', 'Garages', 'Rooms', 'Bedrooms', 'Age',
                            'CtorType', 'AdrType', 'Firefighter'])
# data = pd.read_csv()
# print(data.head())
df = data[
    ['LocalPrice', 'Bathrooms', 'SiteSize', 'LivingSize', 'Garages', 'Rooms', 'Bedrooms', 'Age', 'CtorType', 'AdrType',
     'Firefighter']]
# print(A)

# Add a column of ones for the bias term.
# We chose 1 because if you multiply one with any value, that value does not change.
df = pd.concat([pd.Series(1, index=df.index, name='00'), df], axis=1)
# print(df.head())


data_x_with_names = df.drop('LocalPrice', axis='columns')  # ignores the output variable only.

# Mapping to array from csv
data_x_model = np.array(
    [data_x_with_names['Bathrooms'][0:22], data_x_with_names['SiteSize'][0:22], data_x_with_names['LivingSize'][0:22],
     data_x_with_names['Garages'][0:22], data_x_with_names['Rooms'][0:22], data_x_with_names['Bedrooms'][0:22],
     data_x_with_names['Age'][0:22], data_x_with_names['CtorType'][0:22], data_x_with_names['AdrType'][0:22],
     data_x_with_names['Firefighter'][0:22]])
# print(data_x_model)
data_y_model = np.array(df.iloc[:, 1][0:22])

data_x_test = np.array(
    [data_x_with_names['Bathrooms'][22:28], data_x_with_names['SiteSize'][22:28],
     data_x_with_names['LivingSize'][22:28],
     data_x_with_names['Garages'][22:28], data_x_with_names['Rooms'][22:28], data_x_with_names['Bedrooms'][22:28],
     data_x_with_names['Age'][22:28], data_x_with_names['CtorType'][22:28], data_x_with_names['AdrType'][22:28],
     data_x_with_names['Firefighter'][22:28]])
# print(data_x_test)
data_y_test = np.array(df.iloc[:, 1][22:28])

# Generate random w, or with 0 for your choice.
# w = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
w = np.random.rand(11)

# Generate random b, or with 0 for your choice.
# b = 0
tmp_b = np.random.rand(1)
b = tmp_b[0]

# Alpha selection
alpha = 0.00001


def deriv_w(which_w: int, w_method: list, data_x_method, data_y_method, b_method) -> float:
    sigma = 0
    for k in range(0, len(data_x_method)):
        sigma += w_method[k] * data_x_method[k]
    ans = np.dot(((sigma + b_method) - data_y_method), data_x_method[which_w]) * 1.0 / len(data_y_method)
    return ans


def deriv_b(w_method: list, data_x_method, data_y_method, b_method):
    sigma = 0
    for k in range(0, len(data_x_method)):
        sigma += w_method[k] * data_x_method[k]
    ans = np.mean(1 * ((sigma + b_method) - data_y_method))
    return ans


# Data iterations:
for iteration in range(10000):
    b -= alpha * deriv_b(w_method=w, data_x_method=data_x_model, data_y_method=data_y_model, b_method=b)
    for i in range(0, len(data_x_model)):
        w[i] -= alpha * deriv_w(which_w=i, w_method=w, data_x_method=data_x_model, data_y_method=data_y_model,
                                b_method=b)
    if iteration % 200 == 0:
        # TODO: print entire w's
        print("it: %d, w0: %.3f, w1: %.3f, w2: %.3f, w3: %.3f, w4: %.3f, w5: %.3f, b: %.3f" % (
            iteration, w[0], w[1], w[2], w[3], w[4], w[5], b))

# TODO: TO MODEL 75% and TEST 25%
# Testing the rows 22-28 ( as part of the 25% left after modeling )
arr_test = np.array([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9], w[10]])  # create the test array vector using the results.
for iter in range(0, 6):
    print(f'Estimated price for House.no{iter} :', )

print("Estimated price for Galaxy S5: ", np.dot(np.array([5, 25]), np.array([w[0], w[1]])) + b)
print("Estimated price for Galaxy S1: ", np.dot(np.array([1, 1]), np.array([w[0], w[1]])) + b)
