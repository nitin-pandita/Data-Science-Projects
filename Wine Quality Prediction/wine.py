import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


wine_data = pd.read_csv('E:\Data Science Projects\Wine Quality Prediction\winequality-red.csv')


print(wine_data.describe())


print(sns.catplot(x='quality', data=wine_data, kind='count'))