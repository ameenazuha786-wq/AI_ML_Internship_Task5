
import pandas as pd

# Load the dataset
df = pd.read_csv("heart.csv")

# Display dataset information
print("Dataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# Separate input features and target
X = df.drop(columns=["target"])
y = df["target"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

print("\nTarget distribution:")
print(y.value_counts())


# Split the dataset into training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)
print("Training target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)


# Train a Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
dt_pred = dt_model.predict(X_test)

# Calculate accuracy
dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Accuracy:", round(dt_accuracy * 100, 2), "%")


# Control overfitting by limiting tree depth
dt_limited = DecisionTreeClassifier(max_depth=4, random_state=42)

dt_limited.fit(X_train, y_train)

limited_pred = dt_limited.predict(X_test)

limited_accuracy = accuracy_score(y_test, limited_pred)

print("\nLimited Decision Tree Accuracy:",
      round(limited_accuracy * 100, 2), "%")

print("Full Tree Depth:", dt_model.get_depth())
print("Limited Tree Depth:", dt_limited.get_depth())


# Visualize the Decision Tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    dt_limited,
    feature_names=X.columns,
    class_names=["No Heart Disease", "Heart Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()


# Visualize the Decision Tree
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))

plot_tree(
    dt_limited,
    feature_names=X.columns,
    class_names=["No Heart Disease", "Heart Disease"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree Visualization")
plt.savefig("decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()


# Train a Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Accuracy:",
      round(rf_accuracy * 100, 2), "%")

# Compare both models
print("\nModel Accuracy Comparison:")
print("Decision Tree:", round(dt_accuracy * 100, 2), "%")
print("Random Forest:", round(rf_accuracy * 100, 2), "%")


# Feature Importance using Random Forest
import pandas as pd
import matplotlib.pyplot as plt

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nTop 10 Important Features:")
print(importance.head(10))

# Plot feature importance
plt.figure(figsize=(10, 6))
importance.head(10).sort_values().plot(kind="barh")
plt.title("Top 10 Important Features - Random Forest")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=300)
plt.show()


# Cross-validation for Decision Tree and Random Forest
from sklearn.model_selection import cross_val_score

dt_cv_scores = cross_val_score(
    dt_model, X, y, cv=5, scoring="accuracy"
)

rf_cv_scores = cross_val_score(
    rf_model, X, y, cv=5, scoring="accuracy"
)

print("\nDecision Tree Cross-Validation Accuracy:",
      round(dt_cv_scores.mean() * 100, 2), "%")

print("Random Forest Cross-Validation Accuracy:",
      round(rf_cv_scores.mean() * 100, 2), "%")
