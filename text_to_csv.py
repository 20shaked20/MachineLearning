import pandas as pd

read_file = pd.read_csv (r'/Users/Shaked/PycharmProjects/LinearRegression/prices.txt')
read_file.to_csv (r'/Users/Shaked/PycharmProjects/LinearRegression/prices.csv', index=None)
