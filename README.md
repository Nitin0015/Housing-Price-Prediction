# **Housing Price Prediction**

This repository contains a Jupyter Notebook that demonstrates how to predict housing prices using machine learning techniques. The dataset used in this project is the Boston Housing Dataset, which provides information about housing prices and various factors affecting them.

---

## **Overview**

Predicting housing prices is a critical task in real estate and urban economics. This project uses machine learning to estimate housing prices based on features such as crime rate, property tax rate, number of rooms, and other neighborhood characteristics. The model implemented in this notebook is based on **XGBoost**, a powerful gradient boosting algorithm.

The dataset includes 13 features describing various attributes of Boston neighborhoods and a target variable representing the median value of owner-occupied homes.

---

## **Dataset**

- **Source**: [Boston Housing Dataset](http://lib.stat.cmu.edu/datasets/boston) (originally from Carnegie Mellon University).
- **Features**:
  - CRIM: Per capita crime rate by town.
  - ZN: Proportion of residential land zoned for lots over 25,000 sq. ft.
  - INDUS: Proportion of non-retail business acres per town.
  - CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise).
  - NOX: Nitric oxide concentration (parts per 10 million).
  - RM: Average number of rooms per dwelling.
  - AGE: Proportion of owner-occupied units built prior to 1940.
  - DIS: Weighted distances to five Boston employment centers.
  - RAD: Index of accessibility to radial highways.
  - TAX: Full-value property tax rate per $10,000.
  - PTRATIO: Pupil-teacher ratio by town.
  - B: $$1000(Bk - 0.63)^2$$, where $$Bk$$ is the proportion of Black residents by town.
  - LSTAT: Percentage of lower status population.
- **Target Variable**:
  - MEDV: Median value of owner-occupied homes in $1000s.

---

## **Project Workflow**

1. **Data Loading**:
   - The dataset is loaded from an online source and preprocessed into a structured format using NumPy and Pandas.

2. **Exploratory Data Analysis (EDA)**:
   - Visualizations using Matplotlib and Seaborn are used to explore relationships between features and the target variable.

3. **Model Training**:
   - The dataset is split into training and testing sets using `train_test_split`.
   - An **XGBoost Regressor** (`XGBRegressor`) is trained to predict housing prices.

4. **Model Evaluation**:
   - Performance metrics such as Mean Absolute Error (MAE) and Mean Squared Error (MSE) are calculated to evaluate the model's accuracy.

---

## **Dependencies**

To run this project, you need the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install these dependencies using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost
```

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/HousingPricePrediction.git
   cd HousingPricePrediction
   ```

2. Open the Jupyter Notebook:
   ```bash
   jupyter notebook Housing-Price-Pred.ipynb
   ```

3. Run all cells in the notebook to execute the code.

---

## **Results**

The XGBoost model provides predictions for housing prices based on the input features. The evaluation metrics (e.g., MAE, MSE) indicate how well the model performs. Further tuning and feature engineering can improve performance.

---

## **Acknowledgments**

- The dataset was sourced from [Carnegie Mellon University's StatLib](http://lib.stat.cmu.edu/datasets/boston).
- Special thanks to the developers of XGBoost for providing a robust machine learning library.

---
