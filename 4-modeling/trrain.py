import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('./4-modeling/combined_all.csv')

# Clean and preprocess the data
# Handling missing values
data = data.dropna(subset=['Orientation'])

# Encoding categorical variables
label_encoder = LabelEncoder()
data['Libellé de la commune'] = label_encoder.fit_transform(data['Libellé de la commune'])
data['Nom Complet'] = label_encoder.fit_transform(data['Nom Complet'])
one_hot_encoder = OneHotEncoder()
sex_encoded = one_hot_encoder.fit_transform(data[['Sexe']])
sex_encoded_df = pd.DataFrame(sex_encoded.toarray(), columns=["Sexe_" + str(i) for i in range(sex_encoded.shape[1])])
data.drop(['Sexe'], axis=1, inplace=True)

# Define the target variable
def categorize_orientation(value):
    if value < -0.5:
        return 'Negative'
    elif value > 0.5:
        return 'Positive'
    else:
        return 'Neutral'

data['Orientation_Cat'] = data['Orientation'].apply(categorize_orientation)
data['Orientation_Cat'] = label_encoder.fit_transform(data['Orientation_Cat'])

# Select features
features = data.drop(['Orientation', 'Orientation_Cat'], axis=1)
target = data['Orientation_Cat']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Calculate accuracy and generate a report
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)

# Plot feature importances (for models that support this, e.g., Random Forest)
# feature_importances = pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
# feature_importances.plot(kind='bar')

plt.show()
