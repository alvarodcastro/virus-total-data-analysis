# VirusTotal Android Malware Analysis

## Project Overview
This project implements a comprehensive analysis system for Android malware samples using VirusTotal data. It processes and analyzes JSON reports from VirusTotal's API, providing multiple analytical perspectives on malware behavior, detection patterns, and relationships.

## Project Structure
```
├── VT_Initial_Insert.py    # Initial data ingestion into MongoDB
├── VT_Analysis.py          # Main analysis and visualization scripts
├── VT_modelling.py         # Machine learning models (PCA and prediction)
├── requirements.txt        # Project dependencies
└── results/               # Generated analysis outputs
```

## Features

### 1. Data Processing and Storage
- MongoDB integration for efficient data storage and querying
- Processes VirusTotal JSON reports for Android malware samples
- Handles complex nested data structures

### 2. Statistical Analysis
- Geographic distribution of malware samples
- Detection rates across antivirus engines
- Temporal analysis of sample submissions
- Tag-based categorization and analysis

### 3. Visualizations
- Interactive geographic visualizations using Folium
- Detection rate distribution plots
- Network graphs of malware relationships
- Time series analysis visualizations

## Key Analysis Components

### Geographical Analysis
- Distribution of samples by country
- Interactive world maps with detection statistics
- Country-specific malware submission patterns

### Antivirus Engine Analysis
- Engine-specific detection rates
- Comparative analysis of engine effectiveness
- Detection pattern analysis using PCA

### Malware Relationships
- Analysis of shared children between samples
- Network visualization of malware relationships
- Classification of file types in malware packages

### Temporal Analysis
- Time-based submission patterns
- Hourly and daily submission trends
- Evolution of detection rates over time

## Technical Details

### Dependencies
- Python 3.x
- MongoDB
- Key Python libraries:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib
  - seaborn
  - folium
  - networkx
  - pymongo

### Database Schema
The project uses MongoDB with the following key collections:
- Main collection storing VirusTotal reports
- Indexed fields for efficient querying
- Structured schema for nested data

### Machine Learning Models
1. Children Prediction Model
   - Random Forest Regressor
   - Feature engineering from sample metadata
   - Performance metrics and validation

## Results Directory Structure
The `results/` directory contains various analysis outputs:
- CSV files with detailed analysis results
- PNG/HTML visualization files
- Statistical summaries
- Model performance reports

## Installation and Usage

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure MongoDB connection in `.env` file:
   ```
   MONGO_URI=your_mongodb_connection_string
   DATABASE_NAME=your_database_name
   COLLECTION_NAME=your_collection_name
   ```
4. Run the analysis pipeline:
   ```bash
   python VT_Initial_Insert.py  # Initial data loading
   python VT_Analysis.py        # Run main analysis
   python VT_modelling.py       # Run ML models
   ```

## Analysis Outputs

### Generated Files
- Detailed CSV reports
- Interactive maps
- Statistical visualizations
- Network graphs
- Model performance metrics

### Key Metrics
- Detection rates
- Geographic distributions
- Temporal patterns
- Malware relationships
- Feature importance
- Model accuracy metrics

## Future Improvements
- Real-time analysis capabilities
- Advanced clustering algorithms
- Deep learning model integration
- API automation
- Enhanced visualization options

## License
This project is licensed under the MIT License - see the LICENSE file for details.
