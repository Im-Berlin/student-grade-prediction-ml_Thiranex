import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================
# 1. LOAD DATASET
# ==========================

df = pd.read_csv("student-por.csv")

print("Dataset Shape:")
print(df.shape)

# ==========================
# 2. CONVERT CATEGORICAL DATA
# ==========================

df = pd.get_dummies(df, drop_first=True)

# ==========================
# 3. FEATURES AND TARGET
# ==========================

X = df.drop("G3", axis=1)
y = df["G3"]

# ==========================
# 4. TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Records:", len(X_train))
print("Testing Records:", len(X_test))

# ==========================
# 5. LINEAR REGRESSION
# ==========================

lr = LinearRegression()

lr.fit(X_train, y_train)

pred_lr = lr.predict(X_test)

print("\n===== Linear Regression =====")
print("MAE:", mean_absolute_error(y_test, pred_lr))
print("RMSE:", mean_squared_error(y_test, pred_lr) ** 0.5)
print("R2 Score:", r2_score(y_test, pred_lr))

# ==========================
# 6. DECISION TREE
# ==========================

dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train, y_train)

pred_dt = dt.predict(X_test)

print("\n===== Decision Tree =====")
print("MAE:", mean_absolute_error(y_test, pred_dt))
print("RMSE:", mean_squared_error(y_test, pred_dt) ** 0.5)
print("R2 Score:", r2_score(y_test, pred_dt))

# ==========================
# 7. RANDOM FOREST
# ==========================

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

print("\n===== Random Forest =====")
print("MAE:", mean_absolute_error(y_test, pred_rf))
print("RMSE:", mean_squared_error(y_test, pred_rf) ** 0.5)
print("R2 Score:", r2_score(y_test, pred_rf))

# ==========================
# 8. MODEL COMPARISON
# ==========================

results = {
    "Linear Regression": r2_score(y_test, pred_lr),
    "Decision Tree": r2_score(y_test, pred_dt),
    "Random Forest": r2_score(y_test, pred_rf)
}

print("\n===== Model Comparison =====")
print(results)

# ==========================
# 9. VISUALIZATION
# ==========================

plt.figure(figsize=(8,5))

plt.bar(
    results.keys(),
    results.values()
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("R2 Score")

plt.savefig("Model_Comparison.png")

plt.show()

print("\nModel comparison graph saved successfully.")


# ==============================
# 10. ACTUAL VS PREDICTED GRAPH
# ==============================

plt.figure(figsize=(6,6))

plt.scatter(y_test, pred_lr)

plt.xlabel("Actual Grades")
plt.ylabel("Predicted Grades")
plt.title("Actual vs Predicted Grades")

plt.savefig("Actual_vs_Predicted.png")

plt.show()

print("Actual vs Predicted graph saved successfully.")



# ==============================================================================