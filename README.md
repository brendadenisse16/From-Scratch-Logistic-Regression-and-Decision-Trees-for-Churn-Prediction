# Customer Churn Prediction using Statistical Learning

## Overview
This project focuses on predicting customer churn for bank customers using statistical learning techniques. Customer retention is crucial for organizational success, especially in the financial sector. The data used in this project comes from an actual bank dataset obtained via [Kaggle](https://www.kaggle.com/).

The goal is to build a predictive model that helps banks identify which customers are likely to churn, allowing them to implement proactive retention strategies.

## Algorithms Used
I implemented two statistical learning algorithms **from scratch** in R, without using any built-in machine learning functions:
1. **Logistic Regression**: A widely used algorithm for binary classification problems.
2. **Decision Trees**: A model that provides high interpretability and is particularly useful for understanding customer behavior.

These models were fully coded from the ground up, showcasing the detailed mathematical understanding and coding skills behind each technique.

## Performance Metrics
To evaluate the models, we used the **Area Under the Receiver Operating Characteristic (AUROC)** curve:
- Logistic Regression performed better overall.
- Decision Trees offered higher interpretability, making them useful for marketing strategy insights.

## Project Structure
- **/BinaryTreeDecision.R**: R script implementing the Decision Tree model from scratch.
- **/LogisticRegression.R**: R script implementing Logistic Regression from scratch.
- **/churn.csv**: The dataset used for training and testing the models.

## Setup and Usage
To run the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/brendadenisse16/Statistical-Learning.git
