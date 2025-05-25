import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pymongo import MongoClient
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import os

# Load environment variables from .env file
load_dotenv()

# MongoDB connection
MONGO_URI = os.getenv('MONGO_URI')
DATABASE_NAME = os.getenv('DATABASE_NAME')
COLLECTION_NAME = os.getenv('COLLECTION_NAME')
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading data from MongoDB...")
client = MongoClient(MONGO_URI)
db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

data = list(collection.find())

def extract_features_from_sample(sample):
    """
    Extracts relevant features from a sample for predicting number of children
    """
    exiftool = sample.get("exiftool", {})
    
    return {
        "size": sample.get("size", None),
        "zip_compressed_size": exiftool.get("ZipCompressedSize", None),
        "zip_required_version": exiftool.get("ZipRequiredVersion", None),
        "num_children": sample.get("num_children", None)
    }

def train_children_predictor(data):
    """
    Trains a Random Forest model to predict the number of children in a sample
    """
    print("Extracting features for children prediction...")
    # Extract features from all samples
    records = [extract_features_from_sample(sample) for sample in data]
    df = pd.DataFrame(records)

    # Replace None with NaN for easier handling of missing values
    df.replace({None: np.nan}, inplace=True)

    print("\nInitial data shape:", df.shape)
    print("Columns in the dataset:", df.columns.tolist())
    print("First few rows of the dataset:")
    print(df.head())
    

    # Check for missing values
    print("Columns with missing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    
    # Basic statistics about children
    print("\nBasic statistics about number of children:")
    print(df['num_children'].describe())
    
    # Plot distribution of number of children
    plt.figure(figsize=(10, 6))
    plt.hist(df['num_children'], bins=50)
    plt.title('Distribution of Number of Children')
    plt.xlabel('Number of Children')
    plt.ylabel('Count')
    plt.savefig(os.path.join(RESULTS_DIR, 'children_distribution.png'))
    plt.close()
    

    df.fillna(0, inplace=True)


    print("Columns with missing values after cleaning:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    
    # Split features and target
    X = df.drop('num_children', axis=1)
    y = df['num_children']
    

    print("\nFeatures used for training:")
    print(X.columns.tolist())
    print("Example feature values:")
    print(X.head(10))

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\nTraining set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Train model
    print("\nTraining Random Forest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Number of Children')
    plt.ylabel('Predicted Number of Children')
    plt.title('Actual vs Predicted Number of Children')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'children_prediction.png'))
    plt.close()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title('Feature Importance for Children Prediction')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'children_feature_importance.png'))
    plt.close()
    
    # Save results
    with open(os.path.join(RESULTS_DIR, 'children_prediction_results.txt'), 'w') as f:
        f.write("Children Number Prediction Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Number of samples: {len(df)}\n")
        f.write(f"Mean Absolute Error: {mae:.2f}\n")
        f.write(f"R² Score: {r2:.4f}\n\n")
        f.write("\nFeature Importance:\n")
        for _, row in feature_importance.iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    print(f"\nChildren prediction results:")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    return model

if __name__ == "__main__":
    # Train children predictor
    print("\n=== Training Children Predictor ===")
    model = train_children_predictor(data)
    