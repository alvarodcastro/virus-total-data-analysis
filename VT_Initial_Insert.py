import pandas as pd
import json
from datetime import datetime
import os
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Retrieve MongoDB connection string from environment variables
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')

if not MONGO_URI or not DATABASE_NAME or not COLLECTION_NAME:
    raise ValueError("Missing MongoDB connection string or database/collection name in environment variables.")

directory = "./VTAndroid"
data = []

dataframe = pd.DataFrame(columns=['vhash', 'scan_date', 'first_seen', 'submission_date', 'children', 'total', 'sha256', 'scans', 'tags' ,'positives', 'permalink', 'submission', 'last_seen'])

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]  # New collection for dataframe

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename), 'r') as f:
            data.append(json.load(f))
            
            ## For dataframe
            vhash = data[-1].get('vhash', None)
            scan_date = data[-1].get('scan_date', None)
            first_seen = data[-1].get('first_seen', None)
            submission_date = data[-1].get('submission', {}).get('date', None)
            children = data[-1].get('additional_info', {}).get('compressedview', {}).get('children', None)
            total = data[-1].get('total', None)
            sha256 = data[-1].get('sha256', None)
            scans = data[-1].get('scans', None)  # Store scans as JSON string
            tags = data[-1].get('tags', None)
            positives = data[-1].get('positives', None)
            permalink = data[-1].get('permalink', None)
            submission = data[-1].get('submission', None)  # Store submission as JSON string
            last_seen = data[-1].get('last_seen', None)
            # Insert the row into MongoDB
            document = {
                'vhash': vhash,
                'scan_date': scan_date,
                'first_seen': first_seen,
                'submission_date': submission_date,
                'children': children,
                'total': total,
                'sha256': sha256,
                'scans': scans,
                'positives': positives,
                'permalink': permalink,
                'submission': submission,
                'last_seen': last_seen
            }
        
            # Insert the document into MongoDB
            collection.insert_one(document)

print("Data inserted into MongoDB successfully.")

# Close the MongoDB connection
#client.close()
