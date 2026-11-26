
import joblib
import pandas as pd

from sklearn.inspection import permutation_importance

# Load the saved model pipeline
model = joblib.load("loan_approval_model.pkl")

# Train a separate evaluation model on training data only
# This keeps the validation data independent.
model.fit(X_train, y_train)

# Calculate permutation importance on validation data
importance = permutation_importance(
    model,
    X_val,
    y_val,
    scoring="f1_macro",
    n_repeats=20,
    random_state=42
)

# Create results table
importance_df = pd.DataFrame({
    "Feature": X_val.columns,
    "Importance": importance.importances_mean,
    "Std": importance.importances_std
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nPERMUTATION FEATURE IMPORTANCE")
print(importance_df.to_string(index=False))
