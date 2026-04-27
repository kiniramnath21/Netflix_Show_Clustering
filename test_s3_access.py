import boto3
import pandas as pd
from io import StringIO

# Initialize the S3 client
s3 = boto3.client('s3')

bucket_name = 'netflix-cluster-data'
file_key = 'netflix_cleaned.csv'  # change to netflix_titles.csv if you want to test that

# Fetch the file object from S3
response = s3.get_object(Bucket=bucket_name, Key=file_key)

# Read the CSV content
csv_data = response['Body'].read().decode('utf-8')

# Load into DataFrame
df = pd.read_csv(StringIO(csv_data))

print("✅ Successfully loaded dataset from S3!")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist()[:10])
