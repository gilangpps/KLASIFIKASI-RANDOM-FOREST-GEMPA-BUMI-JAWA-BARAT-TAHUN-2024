import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'D:\GILANG\BEX LABORATORY\DEC 2024\RANDOM FOREST - KLASIFIKASI GEMPA JAWA BARAT 2024/katalog_gempa.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataset to understand its structure
data.head()

# Data cleaning and categorization
# Drop unnecessary columns and handle NaN values
data_cleaned = data[['lat', 'lon', 'depth', 'mag']].dropna()

# Categorize magnitude into 'Small', 'Medium', 'Large'
def categorize_magnitude(mag):
    if mag < 4:
        return 'Small'
    elif 4 <= mag < 6:
        return 'Medium'
    else:
        return 'Large'

data_cleaned['mag_category'] = data_cleaned['mag'].apply(categorize_magnitude)

# Display the distribution of categories
data_cleaned['mag_category'].value_counts()

# Prepare features and target variable
X = data_cleaned[['lat', 'lon', 'depth']]
y = data_cleaned['mag_category']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42, stratify=y)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Display classification report
print(class_report)

# Add visualization for prediction distribution
plt.figure(figsize=(8, 6))

# Distribution of actual categories
sns.histplot(y_test, label='Actual', color='blue', alpha=0.6, bins=3, stat="density")

# Distribution of predicted categories
sns.histplot(y_pred, label='Predicted', color='orange', alpha=0.6, bins=3, stat="density")

plt.title('Distribution of Actual vs Predicted Categories')
plt.xlabel('Magnitude Category')
plt.ylabel('Density')
plt.legend()
plt.show()

