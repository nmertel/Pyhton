import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the Breast Cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize (normalize) the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.45, random_state=42)

# Define the KNN model
knn = KNeighborsClassifier()

# Hyperparameter tuning using Grid Search
param_grid = {'n_neighbors': [3, 5, 7, 9]}
grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print accuracy values for each n_neighbors
print("Grid Search Results:")
for n_neighbors, accuracy in zip(param_grid['n_neighbors'], grid_search.cv_results_['mean_test_score']):
    print(f"n_neighbors={n_neighbors}: Accuracy={accuracy:.6f}")

# Get the best model
best_knn_model = grid_search.best_estimator_

# Make predictions on the test data
y_pred = best_knn_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"\nBest n_neighbors: {best_knn_model.n_neighbors}")
print(f"Accuracy: {accuracy:.2f}")

# Classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# # Save the model using pickle
# with open('knn_model_breast_cancer.pkl', 'wb') as model_file:
#     pickle.dump(best_knn_model, model_file)

# print("K-Nearest Neighbors model successfully saved.")