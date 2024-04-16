import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data_path = './4-modeling/elections_33_clean.csv'  # Update this path to your actual CSV file location
data = pd.read_csv(data_path)

# Remove columns with excessive missing values
threshold = 0.5 * len(data)
data_clean = data.dropna(thresh=threshold, axis=1)

# Fill remaining missing values with the median for numeric columns
for column in data_clean.select_dtypes(include=[np.number]).columns:
    median_value = data_clean[column].median()
    data_clean[column].fillna(median_value, inplace=True)

# Handle missing 'Orientation' column
if 'Orientation' not in data_clean.columns:
    raise ValueError("The column 'Orientation' is missing from the dataset.")
data_clean = data_clean.dropna(subset=['Orientation'])  # Removing rows where 'Orientation' is NaN

# Define features based on the columns available in your dataset
features = ['Pourcentage_Blancs_et_nuls', 'Pourcentage_Abstentions', 'Pourcentage_Votants', 'Voix'] + \
           [col for col in data_clean.columns if 'Age' in col or 'Prof' in col]

# Geographic code and commune name columns
geo_code_column = 'Code de la commune'
commune_name_column = 'Libellé de la commune'

# Preparing the dataset for modeling
X = data_clean[features]
y = data_clean['Orientation']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize and train RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and calculate MSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Predicting orientations for all data
all_predictions = model.predict(X_scaled)
current_year = 2023  # Example current year
future_years = [current_year + i for i in range(1, 4)]  # Next three years

# Create DataFrame for each future year and concatenate them
future_predictions_dfs = []
for year in future_years:
    df = pd.DataFrame({
        geo_code_column: data_clean[geo_code_column],
        commune_name_column: data_clean[commune_name_column],
        'Year': year,
        'Orientation prédite': all_predictions
    })
    future_predictions_dfs.append(df)

final_predictions_df = pd.concat(future_predictions_dfs, ignore_index=True)

# Save to CSV
output_path = './4-modeling/predicted_orientations_future.csv'  # Define your desired output path for the CSV
final_predictions_df.to_csv(output_path, index=False)
print(f"Future predictions saved to {output_path}")

# Print a preview of the DataFrame
print(final_predictions_df.head())
