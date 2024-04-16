from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
# Select relevant features based on your dataset's structure
selected_columns = [
    'Pourcentage_Blancs_et_nuls', 'Pourcentage_Abstentions', 'Pourcentage_Votants',
    'Pop Hommes 30-44 ans (princ)', 'Pop Femmes 75-89 ans (princ)', 'Voix'
] + [col for col in data_clean.columns if 'Age' in col or 'Prof' in col]

# Prepare the DataFrame for modeling
model_data = data_clean[selected_columns + ['Orientation']].dropna()

# Preparing features and target variable
X = model_data.drop('Orientation', axis=1)
y = model_data['Orientation']

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Building and training the random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predicting on the test set
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Predict orientations for all entries
all_predictions = rf_model.predict(X_scaled)

# Generating a DataFrame with geographic codes and predicted orientations
predictions_df = pd.DataFrame({
    'Code géographique': data_clean['Code géographique'],
    'Orientation prédite': all_predictions
})
print(predictions_df.head())
