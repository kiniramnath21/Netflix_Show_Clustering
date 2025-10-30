import pandas as pd
print("🔹 Starting EDA script...")

# Load dataset
df = pd.read_csv("data/netflix_titles.csv")

# Basic info
print("Shape:", df.shape)
print("\nColumns:\n", df.columns)
print("\nMissing values:\n", df.isnull().sum())
print("\nData types:\n", df.dtypes)

# Quick look
print("\nSample rows:")
print(df.head())

# Drop duplicates
df = df.drop_duplicates()

# Clean up missing values
df['country'] = df['country'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')
df['duration'] = df['duration'].fillna('Unknown')

# Save cleaned dataset
df.to_csv("data/netflix_cleaned.csv", index=False)
print("\n✅ Cleaned dataset saved to data/netflix_cleaned.csv")
