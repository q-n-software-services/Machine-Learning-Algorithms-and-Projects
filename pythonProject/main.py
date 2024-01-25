import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initializations
weight = 72.12      # In Kilograms exact
height = 5.75       # In feet and decimals obtained by dividing inches by 12
body_type = 'Asian'     # Regional identification for the body type
exercise_time = 12.12   # In Minutes and decimal obtained from dividing seconds by 60
exercise_days_streak = 5    # Number of days for which the consistency is achieved in exercise/workout
food_intake = 500.12        # Number of Calories, intake of food
exhaustion_level = 0.12     # In Percentage i.e. out of 1
physical_condition = 'Male Teenager'    # Gender and age group


# importing the Dataset
housing_data = pd.read_csv("E:\\Machine Learning\\Pandas\\Mohib_indexFalse.csv")
X = housing_data.iloc[:, :-1].values
y = housing_data.iloc[:, 1].values
print(housing_data)
print(X)
print(y)

