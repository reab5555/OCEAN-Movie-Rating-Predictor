import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# Load the data
data = pd.read_csv('movies_big_five.csv')

# Select the necessary columns as features
features = ['ope', 'con', 'ext', 'agr', 'neu']
X = data[features]
# Select the target
target = 'rating'
y = data[target]

# Define the models to compare
models = {
    'knn': KNeighborsRegressor(metric='euclidean'),
    'randomforest': RandomForestRegressor(random_state=1),
    'gradientboosting': GradientBoostingRegressor(random_state=1),
    'svr': SVR()
}

# Define the parameter grids for each model
param_grids = {
    'knn': {'knn__n_neighbors': range(1, 100)},
    'randomforest': {'randomforest__n_estimators': [10, 50, 100, 125, 150], 'randomforest__max_depth': [None, 1, 5, 10, 20]},
    'gradientboosting': {'gradientboosting__n_estimators': [10, 50, 100, 200, 400, 600, 700], 'gradientboosting__learning_rate': [0.001, 0.0025, 0.005, 0.01, 0.1]},
    'svr': {'svr__C': [0.1, 1, 10, 25], 'svr__kernel': ['linear', 'poly', 'rbf']}
}

n_folds = 7

# Initialize results dictionary
results = {}

for model_name, model in tqdm(models.items(), desc="Model Evaluation"):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])

    param_grid = param_grids[model_name]

    grid_search = GridSearchCV(pipeline, param_grid, cv=n_folds, scoring='neg_mean_squared_error')
    grid_search.fit(X, y)

    best_params = grid_search.best_estimator_

    # Perform cross-validation with the best model
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    r2_scores = []
    rmse_scores = []
    mape_scores = []
    y_pred_combined = np.zeros_like(y)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        best_params.fit(X_train, y_train)
        y_pred_fold = best_params.predict(X_test)
        y_pred_combined[test_index] = y_pred_fold
        r2_fold = r2_score(y_test, y_pred_fold)
        rmse_fold = np.sqrt(mean_squared_error(y_test, y_pred_fold))
        mape_fold = mean_absolute_percentage_error(y_test, y_pred_fold)
        r2_scores.append(r2_fold)
        rmse_scores.append(rmse_fold)
        mape_scores.append(mape_fold)

    # Calculate overall R-squared using the combined predictions
    average_r2 = np.mean(r2_scores)
    average_rmse = np.mean(rmse_scores)
    average_mape = np.mean(mape_scores)

    optimal_params = grid_search.best_params_

    results[model_name] = {
        'optimal_params': optimal_params,
        'average_r2': average_r2,
        'average_rmse': average_rmse,
        'average_mape': average_mape,
        'y_pred_combined': y_pred_combined
    }

# Print the results
for model_name, metrics in results.items():
    print(f'Model: {model_name}')
    print(f'Optimal parameters: {metrics["optimal_params"]}')
    print(f'Average RMSE: {metrics["average_rmse"]:.3f}')
    print(f'Average MAPE: {metrics["average_mape"] * 100:.3f}%')
    print(f'Average R-Squared across folds: {metrics["average_r2"]:.3f}')
    print()

# Identify the best model based on R-squared
best_model_name = max(results, key=lambda k: results[k]['average_r2'])
best_model_metrics = results[best_model_name]
y_pred_combined = best_model_metrics['y_pred_combined']

plt.figure(figsize=(7, 6))

# Scatter plot for cross-validation predictions
sns.scatterplot(x=y, y=y_pred_combined, color='blue')
sns.regplot(x=y, y=y_pred_combined, scatter=False, color='red')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.title(f'Total Predictions vs Actual (K-Folds: {n_folds}) | Best Model: {best_model_name}')

# Display the metrics on the graph
plt.text(0.05, 0.9, f'R-Squared: {best_model_metrics["average_r2"]:.3f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.85, f'RMSE: {best_model_metrics["average_rmse"]:.3f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.05, 0.8, f'MAPE: {best_model_metrics["average_mape"] * 100:.3f}%', transform=plt.gca().transAxes, fontsize=12)

plt.tight_layout()
plt.show()
