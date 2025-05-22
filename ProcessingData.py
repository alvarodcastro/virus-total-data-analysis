import pandas as pd
import json
from datetime import datetime
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import matplotlib.pyplot as plt
# Load environment variables from a .env file
load_dotenv()

# Retrieve MongoDB connection string from environment variables
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

if not MONGO_URI or not DATABASE_NAME or not COLLECTION_NAME:
    raise ValueError("Missing MongoDB connection string or database/collection name in environment variables.")

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]  # New collection for dataframe

def find_documents_by_field(field_name, field_value):
    query = {field_name: field_value}
    results = collection.find(query)
    return list(results)

def to_dataframe(results):
    """
    Converts a list of MongoDB documents to a pandas DataFrame.
    """
    if not results:
        return pd.DataFrame()  # Return an empty DataFrame if no results
    # Normalize MongoDB documents into a pandas DataFrame
    return pd.json_normalize(results)

def main():
    # Example usage
    field_name = "vhash"
    field_value = "cae9663cc552d87c5379e5692f09b5c3"
    documents = find_documents_by_field(field_name, field_value)
    dataframe = to_dataframe(documents)
    print(dataframe.get('scans.Ikarus.version', None)) 

if __name__ == "__main__":
    main()