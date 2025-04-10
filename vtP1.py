import pandas as pd
import json
from datetime import datetime
import os
import matplotlib.pyplot as plt

directory = "./VTAndroid/"
data = []

dataframe1 = pd.DataFrame(columns=['vhash', 'community_reputation', 'first_seen', 'submission_date', 'submission_country'])
dataframe2 = pd.DataFrame(columns=['vhash', 'detected', 'version', 'result', 'update'])

for filename in os.listdir(directory):
    if filename.endswith(".json"):
        with open(os.path.join(directory, filename), 'r') as f:
            data.append(json.load(f))

            ## For dataframe 1
            vhash = data[-1]['vhash']
            community_reputation = data[-1]['community_reputation']
            first_seen = data[-1]['first_seen']
            submission_date = data[-1]['submission']['date']
            submission_country = data[-1]['submission']['submitter_country']
            dataframe1 = dataframe1._append({
                'vhash': vhash,
                'community_reputation': community_reputation,
                'first_seen': first_seen,
                'submission_date': submission_date,
                'submission_country': submission_country
            }, ignore_index=True)

            ## For dataframe 2
            vhash = data[-1]['vhash']
            for key, value in data[-1]['scans'].items():
                detected = value['detected']
                version = value['version']
                result = value['result']
                update = value['update']
                # New dataframe for each scan
                newRow = pd.DataFrame({
                    'vhash': vhash,
                    'detected': detected,
                    'version': version,
                    'result': result,
                    'update': update
                }, index=[key])
                # Concat new row to dataframe2
                dataframe2 = pd.concat([dataframe2, newRow], ignore_index=False)

## For dataframe 3
dataframe3 = pd.merge(dataframe1, dataframe2, on='vhash')
# Count the valid results by country
grouped = dataframe3.groupby('submission_country')['result'].count().reset_index()
# Filter the results that are less than 10000
filtered = grouped[grouped['result'] < 10000].sort_values(by='result', ascending=False)


plt.figure(figsize=(10, 6))
plt.bar(filtered['submission_country'], filtered['result'], color="red", width=0.4)
plt.xlabel('Country')
plt.ylabel('Count')
plt.title('Count by country')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('results.png')
plt.show()

# Results
print("############# Dataframe1 #############")
print("Size:", dataframe1.shape)
print(dataframe1.head())
print(dataframe1.info())
print("Saving dataframe to file...")
dataframe1.to_csv('dataframe1.csv', index=False)
print("#######################################")

print("############# Dataframe2 #############")
print("Size:", dataframe2.shape)
print(dataframe2.head())
print(dataframe2.info())
print("Saving dataframe to file...")
dataframe2.to_csv('dataframe2.csv', index=True, index_label='antivirus')
print("#######################################")

print("############# Dataframe3 #############")
print("Size:", dataframe3.shape)
print(dataframe3.head())
print(dataframe3.info())
print("Saving dataframe to file...")
dataframe3.to_csv('dataframe3.csv', index=False)
print("#######################################")

