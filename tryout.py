import pandas as pd

# Example DataFrame
data = {
    'col1': ['A', 'B', 'C', 'A', 'B'],  # First column with strings
    'col2': [10, 20, 30, 40, 50],
    'col3': [100, 200, 300, 400, 500]
}
df = pd.DataFrame(data)

# Step 1: One-hot encode the first column (strings)
encoded_columns = pd.get_dummies(df['Activity'], prefix='Activity')  # Adds prefix to avoid confusion

# Step 2: Drop the original column and concatenate the encoded columns
df = pd.concat([encoded_columns, df.drop(columns=['Activity'])], axis=1)

# Step 3: Display the final DataFrame
print(df)
