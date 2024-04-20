import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

import logging

logging.basicConfig(
    filename="output.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


# tesk 1

# Load the raw dataset
df = pd.read_csv("test.csv")

## 1. Total In-flight Service Score
df["Total_Inflight_Service_Score"] = df[
    [
        "Inflight wifi service",
        "Seat comfort",
        "Inflight entertainment",
        "On-board service",
        "Leg room service",
        "Cleanliness",
    ]
].sum(axis=1)

# 2. Total Ground Service Score
df["Total_Ground_Service_Score"] = df[
    [
        "Ease of Online booking",
        "Gate location",
        "Food and drink",
        "Online boarding",
        "Baggage handling",
        "Checkin service",
    ]
].sum(axis=1)

# 3. Total Delay Score
df["Total_Delay_Score"] = (
    df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
)

# 4. Travel Distance Category
df["Travel_Distance_Category"] = pd.cut(
    df["Flight Distance"],
    bins=[0, 1000, 3000, float("inf")],
    labels=["Short", "Medium", "Long"],
)

# 5. Overall Satisfaction Score
df["Overall_Satisfaction_Score"] = (
    df["Total_Inflight_Service_Score"]
    + df["Total_Ground_Service_Score"]
    - df["Total_Delay_Score"]
)

# 6. Time of Day
df["Time_of_Day"] = pd.cut(
    df["Departure/Arrival time convenient"],
    bins=[0, 6, 12, 18, 24],
    labels=["Night", "Morning", "Afternoon", "Evening"],
)

# Display the updated dataset
logging.info("The updated dataset is: \n" + str(df.head()))


# Task2

# Dimensions
logging.info("Dimensions of the dataset: \n" + str(df.shape))

# Data Types
logging.info("Data types of features: \n" + str(df.dtypes))


# Summary Statistics
logging.info("Summary statistics for numerical features: \n" + str(df.describe()))

# Unique Values
logging.info(
    "Number of unique values for categorical features: \n"
    + str(df.select_dtypes(include="object").nunique())
)


# Task3

# Separate features (X) and target variable (y)
X = df.drop("satisfaction", axis=1)  # Features
y = df["satisfaction"]  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define preprocessing steps for numerical and categorical features
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X.select_dtypes(include=["object"]).columns

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="mean"),
        ),  # Handle missing values with mean imputation
        ("scaler", StandardScaler()),  # Scale numerical features
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent"),
        ),  # Handle missing values with most frequent imputation
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
        ),  # One-hot encode categorical features
    ]
)

# Combine preprocessing steps for numerical and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Define a pipeline with preprocessing and modeling steps
pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

# Fit the pipeline to the training data
pipeline.fit(X_train)

# Transform the training and testing data
X_train_processed = pipeline.transform(X_train)
X_test_processed = pipeline.transform(X_test)

# Convert the transformed data back to DataFrame
numeric_feature_names = (
    pipeline.named_steps["preprocessor"].transformers_[0][2].tolist()
)
categorical_feature_names = (
    pipeline.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_features)
    .tolist()
)

X_train_processed_df = pd.DataFrame(
    X_train_processed, columns=numeric_feature_names + categorical_feature_names
)
X_test_processed_df = pd.DataFrame(
    X_test_processed, columns=numeric_feature_names + categorical_feature_names
)

# Task 4

df_processed = df.copy()  # Placeholder example

# Describe the dataset and its features after processing
logging.info("Dimensions of the dataset after processing: \n" + str(df_processed.shape))
# Dimensions
logging.info("Data types of features after processing: \n" + str(df_processed.dtypes))
# shape
logging.info("Dimensions of the dataset after processing: \n" + str(df_processed.shape))
# Data Types
# Summary Statistics
logging.info(
    "Summary statistics for numerical features after processing: \n"
    + str(df_processed.describe())
)


# Unique Values
logging.info(
    "Number of unique values for categorical features after processing: \n"
    + str(df_processed.select_dtypes(include="object").nunique())
)

# Task 5

# Load the dataset
df = pd.read_csv("test.csv")

# Select interesting features for analysis (e.g., numerical features)
interesting_features = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Seat comfort",
    "Ease of Online booking",
    "Food and drink",
    "Online boarding",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# Visualize distributions of interesting features
for feature in interesting_features:
    plt.figure(figsize=(8, 5))
    plt.hist(df[feature], bins=20, color="skyblue", edgecolor="black")
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"output/{feature}_distribution.png", format="png")
    logging.info(
        f"Saved distribution plot for {feature} as '{feature}_distribution.png'"
    )
    plt.show()

# Assess central tendency and variability
summary_statistics = df[interesting_features].describe()
logging.info(
    "Summary statistics for interesting features: \n" + str(summary_statistics)
)
missing_values = df[interesting_features].isnull().sum()
logging.info("Missing values for interesting features: \n" + str(missing_values))

# Calculate mean and standard deviation
means = df[interesting_features].mean()
std_devs = df[interesting_features].std()
logging.info("Mean Values: \n" + str(means))
logging.info("Standard Deviations: \n" + str(std_devs))

df["Arrival Delay in Minutes"].fillna(means["Arrival Delay in Minutes"], inplace=True)

# Detect skewness and outliers
skewness = df[interesting_features].skew()
outliers = df[interesting_features][
    df[interesting_features].apply(lambda x: (x - x.mean()).abs() > 2 * x.std())
]
logging.info("Skewness: \n" + str(skewness))
logging.info("Outliers: \n" + str(outliers))

# Explore relationships between features (e.g., correlation)
correlation_matrix = df[interesting_features].corr()
logging.info("Correlation Matrix: \n" + str(correlation_matrix))

# Task 6 - in the pdf

# Task 7

# Assumption 1: Positive correlation between 'Inflight wifi service' and 'Online boarding' ratings
correlation_wifi_boarding = df["Inflight wifi service"].corr(df["Online boarding"])
logging.info(
    f"Correlation between Inflight wifi service and Online boarding: {correlation_wifi_boarding}"
)


# Assumption 2: Higher 'Inflight entertainment' ratings for younger passengers
age_groups = pd.cut(
    df["Age"], bins=[0, 18, 30, 50, np.inf], labels=["<18", "18-30", "30-50", "50+"]
)
df["Age Group"] = age_groups
sns.boxplot(x="Age Group", y="Inflight entertainment", data=df)
plt.title("Inflight Entertainment Ratings by Age Group")
plt.xlabel("Age Group")
plt.ylabel("Inflight Entertainment Rating")
plt.grid(False)
plt.savefig("output/Inflight_Entertainment_Age_Group.png", format="png")
logging.info(
    "Saved Inflight Entertainment Ratings by Age Group plot as 'Inflight_Entertainment_Age_Group.png'"
)
plt.show()

# Assumption 3: Higher 'Inflight wifi service' ratings for business travelers
mean_wifi_business = df[df["Customer Type"] == "Business travel"][
    "Inflight wifi service"
].mean()
mean_wifi_personal = df[df["Customer Type"] == "Personal Travel"][
    "Inflight wifi service"
].mean()
logging.info(
    f"Mean Inflight wifi service rating for Business travelers: {mean_wifi_business}"
)
logging.info(
    f"Mean Inflight wifi service rating for Personal travelers: {mean_wifi_personal}"
)

# Assumption 4: Higher 'Seat comfort' ratings for higher class passengers
sns.boxplot(x="Class", y="Seat comfort", data=df)
plt.title("Seat Comfort Ratings by Class")
plt.xlabel("Class")
plt.ylabel("Seat Comfort Rating")
plt.grid(False)
plt.savefig("output/Seat_Comfort_Class.png", format="png")
logging.info("Saved Seat Comfort Ratings by Class plot as 'Seat_Comfort_Class.png'")
plt.show()

# Assumption 5: Positive correlation between 'Flight Distance' and 'Inflight entertainment' ratings
correlation_distance_entertainment = df["Flight Distance"].corr(
    df["Inflight entertainment"]
)
logging.info(
    f"Correlation between Flight Distance and Inflight entertainment: {correlation_distance_entertainment}"
)


# Task 8

# Select numerical features for correlation analysis
numerical_features = df.select_dtypes(include=["int64", "float64"])

# Calculate the correlation matrix
correlation_matrix = numerical_features.corr()
# Plot the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.savefig("output/Correlation_Matrix.png", format="png")
logging.info("Correlation Matrix plot saved as 'Correlation_Matrix.png'")
plt.show()

# Task 9

# Use feature selection techniques or model-based analysis to find the best features influencing the target variable 'satisfaction'
# 1. Feature Selection using ANOVA (Analysis of Variance)
# ANOVA is a statistical method used to test differences between two or more means.
# Here, we use it to determine the most significant features in the dataset.
X = df.drop(columns=["satisfaction"])
y = df["satisfaction"]
y = LabelEncoder().fit_transform(y)
X = pd.get_dummies(X, drop_first=True)

# Apply ANOVA to select the top k features
selector = SelectKBest(score_func=f_classif, k="all")
selector.fit(X, y)

# Get the ANOVA scores and corresponding p-values
# The F-Score represents the ratio of variation between the sample means to the variation within the samples.
# A higher F-Score indicates a more significant feature.
anova_results = pd.DataFrame(
    {
        "Feature": X.columns,
        "ANOVA F-Score": selector.scores_,
        "p-value": selector.pvalues_,
    }
)
anova_results.sort_values(by="ANOVA F-Score", ascending=False, inplace=True)
logging.info("ANOVA Results: \n" + str(anova_results))
top_anova_results = anova_results.nlargest(10, "ANOVA F-Score")
colors = plt.cm.viridis(np.linspace(0, 1, len(top_anova_results)))
# Visualize the ANOVA F-scores
plt.figure(figsize=(12, 8))
plt.barh(
    top_anova_results["Feature"],
    top_anova_results["ANOVA F-Score"],
    color=colors,
)
plt.xlabel("ANOVA F-Score")
plt.ylabel("Feature")
plt.title("ANOVA F-Scores for Feature Selection")
plt.grid(axis="x", linestyle="--", alpha=0.7)

plt.savefig("output/ANOVA_FScore.png", format="png")
logging.info("ANOVA F-Score plot saved as 'ANOVA_FScore.png'")
plt.show()


# 2. Model-Based Analysis using RandomForestClassifier
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train, y_train)

# Get feature importances from the trained model
feature_importances = pd.DataFrame(
    {"Feature": X.columns, "Importance": rf_classifier.feature_importances_}
)
feature_importances.sort_values(by="Importance", ascending=False, inplace=True)
logging.info(
    "Feature Importances from RandomForestClassifier: \n" + str(feature_importances)
)

top_feature_importances = feature_importances.nlargest(10, "Importance")
colors = plt.cm.viridis(np.linspace(0, 1, len(top_feature_importances)))

# Visualize the feature importances
plt.figure(figsize=(12, 8))
plt.barh(
    top_feature_importances["Feature"],
    top_feature_importances["Importance"],
    color=colors,
)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances from RandomForestClassifier")
plt.grid(axis="x", linestyle="--", alpha=0.7)

# Add importance values on the bars
for index, value in enumerate(top_feature_importances["Importance"]):
    plt.text(
        value, index, str(round(value, 4)), va="center", fontsize=10, color="black"
    )
plt.savefig("output/Feature_Importance.png", format="png")
logging.info("Feature Importance plot saved as 'Feature_Importance.png'")
plt.show()
