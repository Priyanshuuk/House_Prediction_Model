# House_Prediction_Model

Learning about Machine Learning, Exploratory Data Analysis (EDA), and Linear Regression through the California Housing dataset.

# California Housing Price Prediction

A machine learning project that predicts **median house values in California** using the California Housing dataset.

This project was created to learn the basic workflow of a supervised machine learning regression problem, including data exploration, data cleaning, feature engineering, categorical encoding, Linear Regression, and model evaluation.

## Project Overview

The goal of this project is to predict the **median house value** of a California housing district using information such as:

- Geographic location
- Median house age
- Total number of rooms
- Total number of bedrooms
- Population
- Number of households
- Median income
- Ocean proximity

The main model used in this project is **Linear Regression**.

## Dataset

The project uses the California Housing dataset.

### Features

| Feature | Description |
|---|---|
| `longitude` | Longitude of the housing district |
| `latitude` | Latitude of the housing district |
| `housing_median_age` | Median age of houses in the district |
| `total_rooms` | Total number of rooms |
| `total_bedrooms` | Total number of bedrooms |
| `population` | Population of the district |
| `households` | Number of households |
| `median_income` | Median income of households |
| `ocean_proximity` | Proximity of the district to the ocean |
| `median_house_value` | Median house value (target variable) |

## Exploratory Data Analysis

I explored the dataset to understand its structure and relationships between different features.

The following were investigated:

- Dataset shape
- Data types
- Missing values
- Statistical information
- Feature distributions
- Categorical variables
- Correlations between numerical features
- Relationship between income and house value
- Geographic distribution of house prices

### Correlation Heatmap

A correlation heatmap was used to understand relationships between numerical features.

One of the strongest relationships observed was between:

```text
median_income
      ↓
median_house_value


This shows that median_income has a strong relationship with median_house_value.

Data Cleaning

The dataset contained missing values in the total_bedrooms column.

The missing rows were removed using:

data.dropna(inplace=True)

After removing the missing values, the dataset contained 20,433 rows.

Train/Test Split

The target variable was separated from the input features:

X = data.drop('median_house_value', axis=1)
y = data['median_house_value']

The data was then split into training and testing sets:

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

The split consists of:

80% training data
20% testing data
Feature Engineering

Additional features were created from the existing data to provide the model with more useful information.

Rooms per Household
train_data['rooms_per_household'] = (
    train_data['total_rooms'] /
    train_data['households']
)
Bedrooms per Room
train_data['bedrooms_per_room'] = (
    train_data['total_bedrooms'] /
    train_data['total_rooms']
)
Population per Household
train_data['population_per_household'] = (
    train_data['population'] /
    train_data['households']
)
Categorical Encoding

The ocean_proximity column contains categorical values such as:

<1H OCEAN
INLAND
ISLAND
NEAR BAY
NEAR OCEAN

Since Linear Regression requires numerical input, One-Hot Encoding was used to convert the categorical values into numerical columns.

train_data = train_data.join(
    pd.get_dummies(
        train_data['ocean_proximity'],
        dtype=int
    )
).drop('ocean_proximity', axis=1)
Linear Regression

The main machine learning algorithm used in this project is Linear Regression.

from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x_train, y_train)

The model learns the relationship between the input features and the target variable, median_house_value.

Model Evaluation

The model was evaluated using:

R² Score
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
from sklearn.metrics import mean_squared_error, r2_score

y_pred = reg.predict(x_test)

r2 = r2_score(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

print("R²:", r2)
print("MSE:", mse)
print("RMSE:", rmse)
R² Score

R² measures how well the model explains the variation in the target variable.

A value closer to 1 generally indicates a better fit.

Mean Squared Error

MSE measures the average squared difference between the actual and predicted values.

Lower MSE indicates better performance.

Root Mean Squared Error

RMSE is the square root of MSE and is expressed in the same units as the target variable.

Lower RMSE indicates better performance.

What I Learned

Through this project, I learned about:

Machine Learning basics
Pandas
NumPy
Data cleaning
Handling missing values
Exploratory Data Analysis
Data visualization
Correlation
Train/Test splitting
Feature engineering
One-Hot Encoding
Linear Regression
Model prediction
R² Score
MSE
RMSE
Technologies Used
Python
Jupyter Notebook
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Project Structure
House_Prediction_Model/
│
├── housing.csv
├── housing_prediction_model.ipynb
└── README.md
Future Improvements
Try Ridge Regression
Try Lasso Regression
Try Random Forest Regression
Use GridSearchCV for hyperparameter tuning
Improve feature engineering
Build a preprocessing pipeline
Compare multiple models
Improve model evaluation
Create predicted vs actual visualizations
Conclusion

This project was created as a learning project to understand the fundamentals of Machine Learning and regression.

The workflow followed was:

Dataset
   ↓
Data Exploration
   ↓
Data Cleaning
   ↓
Train/Test Split
   ↓
EDA
   ↓
Feature Engineering
   ↓
Categorical Encoding
   ↓
Linear Regression
   ↓
Model Evaluation

This project helped me understand how to take a dataset, explore it, prepare it for Machine Learning, train a regression model, and evaluate its predictions.
