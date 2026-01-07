import pandas as pd
from sklearn.datasets import load_breast_cancer
from pathlib import Path

# Load the breast cancer Wisconsin dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Assert that there are no missing values in any column
assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
print("✓ No missing values found in the dataset")

# Check the dtypes of all columns
print("\nData types:")
print(df.dtypes)
print(f"\nDataset shape: {df.shape}")
print(f"Number of features: {len(data.feature_names)}")

# Save the dataset as parquet in the input folder
output_path = Path("input/breast_cancer.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(output_path, index=False)
print(f"\n✓ Dataset saved to {output_path}")