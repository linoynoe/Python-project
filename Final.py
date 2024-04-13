import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#tesk 1

# Load the raw dataset
df = pd.read_csv("test.csv")

## 1. Total In-flight Service Score
df['Total_Inflight_Service_Score'] = df[['Inflight wifi service', 'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service', 'Cleanliness']].sum(axis=1)

# 2. Total Ground Service Score
df['Total_Ground_Service_Score'] = df[['Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding', 'Baggage handling', 'Checkin service']].sum(axis=1)

# 3. Total Delay Score
df['Total_Delay_Score'] = df['Departure Delay in Minutes'] + df['Arrival Delay in Minutes']

# 4. Travel Distance Category
df['Travel_Distance_Category'] = pd.cut(df['Flight Distance'], bins=[0, 1000, 3000, float('inf')], labels=['Short', 'Medium', 'Long'])

# 5. Overall Satisfaction Score
df['Overall_Satisfaction_Score'] = df['Total_Inflight_Service_Score'] + df['Total_Ground_Service_Score'] - df['Total_Delay_Score']

# 6. Time of Day
df['Time_of_Day'] = pd.cut(df['Departure/Arrival time convenient'], bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

# Display the updated dataset
print(df.head())

#tesk 2

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Check the data types of each column
print("\nData types of each column:")
print(df.dtypes)

# Summary statistics for numerical columns
print("\nSummary statistics for numerical columns:")
print(df.describe())

# Summary of categorical columns
print("\nSummary of categorical columns:")
print(df.describe(include=['object']))

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Unique values in categorical columns
print("\nUnique values in categorical columns:")
for col in df.select_dtypes(include=['object']):
    print(f"{col}: {df[col].unique()}")

# Target variable distribution
print("\nTarget variable distribution:")
print(df['satisfaction'].value_counts())

#task 4

interesting_numerical_features = ['Flight Distance', 'Inflight wifi service', 'Online boarding', 'Seat comfort']

for feature in interesting_numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True, bins=20, color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Describe the distribution of interesting categorical features
interesting_categorical_features = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']

for feature in interesting_categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=df, palette='viridis')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Explore relationships between interesting features
sns.pairplot(df[interesting_numerical_features], diag_kind='kde', corner=True)
plt.show()

# Visualize correlation matrix for numerical features
numerical_features_corr = df[interesting_numerical_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(numerical_features_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix for Numerical Features")
plt.show()

#task 5

#Describe the distribution of numerical features
numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[feature], kde=True, bins=20)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Describe the distribution of categorical features
categorical_features = df.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=feature, data=df, palette='viridis')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

# Explore relationships between features
sns.pairplot(df[numerical_features])
plt.show()

# Visualize correlation matrix
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

#Task 8

correlation_matrix = df.corr()

# Visualize correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Identify interesting correlations (absolute correlation > 0.5)
interesting_correlations = correlation_matrix[abs(correlation_matrix) > 0.5]

# Plot interesting correlations
for i in range(len(interesting_correlations.columns)):
    for j in range(i+1, len(interesting_correlations.columns)):
        if abs(interesting_correlations.iloc[i, j]) > 0.5:
            sns.scatterplot(x=interesting_correlations.columns[i], y=interesting_correlations.columns[j], data=df)
            plt.title(f"{interesting_correlations.columns[i]} vs {interesting_correlations.columns[j]}")
            plt.xlabel(interesting_correlations.columns[i])
            plt.ylabel(interesting_correlations.columns[j])
            plt.show()

