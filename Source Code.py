import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from a URL
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
data = pd.read_csv(url)

# Save the dataset to a local file
data.to_csv("titanic_dataset.csv", index=False)
print("Dataset saved as titanic_dataset.csv")

# Overview of the dataset
print(data.head())
print(data.info())

# Summary statistics
print(data.describe())

# Check for missing values
print(data.isnull().sum())

# Data visualization

# Survival count
plt.figure(figsize=(6, 4))
sns.countplot(x='Survived', data=data)
plt.title('Survival Count')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()

# Survival by class
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Class')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Age distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['Age'].dropna(), kde=True)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Fare distribution
plt.figure(figsize=(6, 4))
sns.histplot(data['Fare'], kde=True)
plt.title('Fare Distribution')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()

# Box plot of age by class
plt.figure(figsize=(6, 4))
sns.boxplot(x='Pclass', y='Age', data=data)
plt.title('Age Distribution by Class')
plt.xlabel('Class')
plt.ylabel('Age')
plt.show()

# Drop or exclude non-numeric columns before computing correlations
numeric_data = data.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_data.corr()

# Heatmap of correlations
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
