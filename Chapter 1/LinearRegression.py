import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
import sklearn
import html

data = pd.read_csv("D:/OneDrive/Documents/101 Online Learning/101 Udemy/machine-learning-101-with-scikit-learn-and-stats-models/S3_L17/real_estate_price_size.csv")
print(data.head(10))
print(data.describe())