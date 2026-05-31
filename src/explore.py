import pandas as pd

# Load the dataset
df = pd.read_csv('../data/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

# Display basic information about the dataset
print("Shape: ", df.shape)

# Show Column name
print("Columns: ", df.columns)

# Show first 5 rows

print("First 5 rows: ")
print(df.head())


# Show what labels (bening or attack) are in the dataset and how many of each
print("Labels: ")
print(df[' Label'].value_counts())
