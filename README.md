# OCEAN-Movie-Rating-Predictor
This project aims to predict movie ratings based on Big Five personality traits.

## Goal
The primary goal of this project is to investigate whether aggregated personality traits of movies can serve as predictors for movie ratings. By comparing different machine learning models, we aim to find the most suitable approach for this prediction task.

This exploration could provide insights into the relationship between personality traits and movie preferences, potentially aiding in the development of personalized recommendation systems.

## Dataset
The dataset used in this project is derived from the MyPersonality dataset, which contains Big Five personality questionnaire responses from about 1000 Facebook users along with the movies they liked. For each movie, there are calculated aggregated mean scores for each of the Big Five personality traits (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) across all users who liked that movie.

The target variable (what we are trying to predict) is the rating of each movie.

## Models
We evaluate multiple machine learning regression models for our predictions.    
Each model undergoes hyperparameter tuning using GridSearchCV and is evaluated using 7-fold cross-validation.

## Results

| Model | Optimal Parameters | Average RMSE | Average MAPE | Average R-Squared |
|-------|-------------------|-------------|-------------|------------------|
| K-Nearest Neighbors (KNN) | 23 neighbors | 0.839 | 10.184% | 0.291 |
| Random Forest | 100 estimators, max depth of 5 | 0.848 | 10.068% | 0.275 |
| Gradient Boosting | 600 estimators, learning rate of 0.005 | 0.856 | 10.251% | 0.261 |
| Support Vector Regression (SVR) | C=1, linear kernel | 0.847 | 10.160% | 0.276 |

### Model Comparison

Among the four models, the K-Nearest Neighbors (KNN) regressor performed the best, achieving the highest R-squared value (0.291) and the lowest RMSE (0.839). This suggests that the KNN model explains approximately 29.1% of the variance in movie ratings based on the aggregated personality traits.

While the predictive power of these models is modest (with R-squared values around 0.3), they do suggest a relationship between aggregated personality traits and movie ratings.

These results highlight the challenging nature of predicting subjective ratings based on personality traits alone. However, they also demonstrate that personality traits do contain some predictive information about movie preferences, which could be valuable in recommendation systems, especially when combined with other features.
