
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("loan_approval_model.pkl")

# Get feature names after preprocessing
feature_names = (
    model.named_steps["preprocessor"].get_feature_names_out()
)

# Get SVM coefficients
coefficients = model.named_steps["model"].coef_[0]

# Create feature importance table
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Importance": abs(coefficients)
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nFEATURE INFLUENCE IN SVM MODEL")
print(importance_df.to_string(index=False))
