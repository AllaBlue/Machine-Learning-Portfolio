# Titanic Survival Prediction with Decision Tree and Random Forest

This folder contains a machine learning notebook that implements **Decision Tree** and **Random Forest** algorithms to predict whether a passenger survived the Titanic disaster. The notebook walks through data cleaning, preprocessing, model training, and performance evaluation for both models. Additionally, the notebook explores optimal hyperparameters for both Decision Tree and Random Forest models.

## Table of Contents

1. [Dataset](#dataset)
2. [Data Preprocessing](#data-preprocessing)
3. [Machine Learning Models](#machine-learning-models)
4. [Performance Evaluation](#performance-evaluation)
5. [Summary of Results](#summary-of-results)

## Dataset

The dataset used in this project is the **Titanic dataset**, which contains various features related to the passengers aboard the Titanic, such as:
- Passenger information (`Age`, `Sex`, `Fare`, etc.)
- The target variable is `Survived` (1 for survived, 0 for not survived).

The dataset undergoes several preprocessing steps to prepare it for training and testing the models.

## Data Preprocessing

The following steps were performed to clean and preprocess the data:
1. **Dropping Irrelevant Features**: Features like `PassengerId`, `Name`, `Ticket`, and `Cabin` were removed as they are irrelevant to the survival prediction.
2. **Handling Missing Values**:
   - Missing values in the `Age` column were replaced with the mean age.
   - Rows with missing values in `Embarked` were dropped.
3. **One-Hot Encoding**: Categorical variables such as `Sex` and `Embarked` were converted to dummy variables using one-hot encoding.
4. **Standardization**: The feature values were standardized using `StandardScaler` to normalize the data.

## Machine Learning Models

The notebook implements and evaluates two models:

### Decision Tree
1. **Initial Decision Tree**: The model is trained and evaluated using the default settings. 
2. **Tree Depth Optimization**: The notebook performs hyperparameter tuning by adjusting the `max_depth` parameter to improve the accuracy of the Decision Tree model.
3. **Final Decision Tree**: A Decision Tree model with optimal depth is trained and evaluated.

### Random Forest
1. **Initial Random Forest**: A basic Random Forest model is trained and evaluated.
2. **Hyperparameter Tuning**: The notebook tunes parameters such as `n_estimators`, `max_features`, and `max_depth` to find the best combination for optimal performance.
3. **Final Random Forest**: The best-performing Random Forest model is trained using the optimal parameters identified during hyperparameter tuning.

## Performance Evaluation

The models are evaluated using various performance metrics, including:
1. **Accuracy**: The percentage of correctly predicted outcomes.
2. **Precision, Recall, and F1-Score**: These metrics are calculated to assess the quality of predictions for each class (survived vs. not survived).
3. **Confusion Matrix**: A confusion matrix is plotted to visualize the model's performance in terms of true positives, true negatives, false positives, and false negatives.
4. **Feature Importance**: The most important features contributing to the predictions are identified and displayed for the Decision Tree model.


## Summary of Results

The Decision Tree and Random Forest models were evaluated using the Titanic dataset, and the following key results were observed:

- **Optimal Decision Tree**: A Decision Tree model with a `max_depth` of 5 achieved significantly higher accuracy than the default model.
- **Random Forest**: The final Random Forest model with tuned hyperparameters (`n_estimators`, `max_features`, `max_depth`) provided the best accuracy on the test data.
- **Feature Importance**: Features like `Fare`, `Age`, and `Pclass` were identified as the most important factors in predicting survival.

The notebook demonstrates how to apply Decision Trees and Random Forests to a classification problem, tune hyperparameters, and interpret the results using performance metrics.
