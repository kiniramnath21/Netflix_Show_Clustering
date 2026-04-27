# import boto3
import pandas as pd
from io import StringIO

# Initialize the S3 client
# s3 = boto3.client('s3')

# bucket_name = 'netflix-cluster-data'
# file_key = 'netflix_cleaned.csv'

# Fetch the file object from S3
# response = s3.get_object(Bucket=bucket_name, Key=file_key)

# Load locally instead
df = pd.read_csv("data/netflix_cleaned.csv")

print("✅ Successfully loaded dataset locally!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:10])