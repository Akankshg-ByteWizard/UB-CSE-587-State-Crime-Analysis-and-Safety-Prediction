# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the dataset from a CSV file
project_df = pd.read_csv(r"Dataset.csv")

# Drop rows with missing values (NaN)
project_df = project_df.dropna()

# Convert data types of columns based on their content
project_df = project_df.convert_dtypes()

# Convert the 'State' column to lowercase for consistency
project_df['State'] = project_df['State'].str.lower()

# Create a copy of the DataFrame for further processing
df = project_df.copy()

# Initialize a label encoder to convert categorical 'State' values to numerical
label_encoder = LabelEncoder()

# Encode the 'State' column to numerical values
df['State'] = label_encoder.fit_transform(df['State'])

# Split the dataset into features (X) and target (y)
X = df[['State', 'Year', 'Data.Population', 'Data.Totals.Property.All', 'Data.Totals.Violent.All']]
y = df['Data.Total.crime']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a list of numerical variables
num_var = ['State', 'Year', 'Data.Population', 'Data.Totals.Property.All', 'Data.Totals.Violent.All']

# Create a numerical data processing pipeline
num_pipeline = Pipeline([
    ('impute_missing', SimpleImputer(strategy='median')),  # Impute missing values with median
    ('standardize_num', StandardScaler())  # Standardize numerical features
])

# Create a column transformer to apply transformations to specific columns
processing_pipeline = ColumnTransformer(transformers=[
    ('proc_numeric', num_pipeline, num_var),  # Apply numerical pipeline to specified columns
])

# Create a Random Forest regression pipeline
rf_pipeline = Pipeline([
    ('data_processing', processing_pipeline),  # Apply data processing
    ('rf', RandomForestRegressor())  # Use Random Forest Regressor for prediction
])

# Fit the Random Forest model to the training data
rf_pipeline.fit(X_train, y_train)

# Define a grid of hyperparameters to search over
param_grid = {
    'rf__n_estimators': [100],
    'rf__max_depth': [None, 10],
    'rf__min_samples_split': [2, 5],
    'rf__min_samples_leaf': [1, 2],
}

# Create a GridSearchCV object to find the best hyperparameters
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the GridSearchCV to the training data
grid_search.fit(X_train, y_train)

# Save the best model to a file using joblib
joblib.dump(grid_search, 'grid_search.pkl')

# Print the best hyperparameters found by GridSearchCV
print("Best Hyperparameters:", grid_search.best_params_)

# Get the best model from GridSearchCV
best_rf_model = grid_search.best_estimator_

# Make predictions on the test set
test_predictions = best_rf_model.predict(X_test)

# Calculate and print the Mean Squared Error on the test set
print("Mean Squared Error on Test Set:", mean_squared_error(y_test, test_predictions))

# Calculate and print the R-squared (Coefficient of Determination) on the test set
print("R-squared on Test Set:", r2_score(y_test, test_predictions) * 100)
