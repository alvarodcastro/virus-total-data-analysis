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

directory = "../VTAndroid/"
data = []

dataframe = pd.DataFrame(columns=['vhash', 'scan_date', 'first_seen', 'submission_date', 'children', 'sha256', 'scans', 'permalink','submission'])

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename), 'r') as f:
            data.append(json.load(f))

            # Connect to MongoDB
            client = MongoClient(MONGO_URI)
            db = client[DATABASE_NAME]
            collection = db[COLLECTION_NAME]  # New collection for dataframe

            ## For dataframe
            vhash = data[-1].get('vhash', None)
            scan_date = data[-1].get('scan_date', None)
            first_seen = data[-1].get('first_seen', None)
            submission_date = data[-1].get('submission', {}).get('date', None)
            children = data[-1].get('additional_info', {}).get('compressedview', {}).get('children', None)
            sha256 = data[-1].get('sha256', None)
            scans = data[-1].get('scans', None)  # Store scans as JSON string
            permalink = data[-1].get('permalink', None)
            submission = data[-1].get('submission', None)  # Store submission as JSON string

            # Insert the row into MongoDB
            document = {
                'vhash': vhash,
                'scan_date': scan_date,
                'first_seen': first_seen,
                'submission_date': submission_date,
                'children': children,
                'sha256': sha256,
                'scans': scans,
                'permalink': permalink,
                'submission': submission
            }
            collection.insert_one(document)

            

