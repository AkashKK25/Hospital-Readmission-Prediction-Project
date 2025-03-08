# Hospital Readmission Prediction Project - Installation and Usage Guide

This guide provides step-by-step instructions for setting up and running the Hospital Readmission Prediction project.

## Prerequisites

- Python 3.8 or higher
- Pip or Conda package manager
- Git (optional, for cloning the repository)
- Minimum 8GB RAM recommended for model training
- Internet connection for downloading the dataset

## Installation

### Option 1: Using pip (Recommended)

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/yourusername/hospital-readmission-project.git
   cd hospital-readmission-project
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   # Using venv
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the project in development mode**:
   ```bash
   pip install -e .
   ```

   This will install all the required dependencies specified in `setup.py`.

### Option 2: Using conda

1. **Clone or download the repository**:
   ```bash
   git clone https://github.com/yourusername/hospital-readmission-project.git
   cd hospital-readmission-project
   ```

2. **Create and activate a conda environment**:
   ```bash
   conda create --name readmission python=3.8
   conda activate readmission
   ```

3. **Install the required packages**:
   ```bash
   conda install -c conda-forge numpy pandas scikit-learn matplotlib seaborn jupyter
   conda install -c conda-forge plotly dash shap xgboost lightgbm imbalanced-learn
   pip install -e .
   ```

## Project Structure

```
hospital_readmission_project/
├── data/                           # Data files (created during execution)
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned and preprocessed data
├── notebooks/                      # Jupyter notebooks
│   └── 1_exploratory_analysis.ipynb    # Initial data exploration
├── src/                            # Source code
│   ├── data/                       # Data processing modules
│   │   ├── acquisition.py          # Data download and import
│   │   └── preprocessing.py        # Data cleaning functions
│   ├── models/                     # Model training and evaluation
│   │   └── train_model.py          # Model training
│   └── analysis/                   # Analysis modules
│       └── business_analysis.py    # Business impact analysis
├── reports/                        # Generated analysis
│   ├── figures/                    # Generated graphics
│   └── Hospital_Readmission_Report.md    # Detailed analysis report
├── dashboard/                      # Interactive dashboard files
│   └── app.py                      # Dashboard application
├── run_analysis.py                 # Main script to run the analysis pipeline
├── requirements.txt                # Dependencies
├── setup.py                        # Setup script
├── README.md                       # Project overview
└── INSTALLATION.md                 # This installation guide
```

## Usage

### Running the Complete Pipeline

To run the entire analysis pipeline (data acquisition, preprocessing, model training, and business analysis):

```bash
python run_analysis.py
```

This script will:
1. Download the diabetes dataset from UCI ML Repository
2. Clean and preprocess the data
3. Train and evaluate predictive models
4. Perform business impact analysis
5. Generate figures and reports

### Command-line Options

The `run_analysis.py` script supports several command-line options:

- `--skip-preprocessing`: Skip the data preprocessing step
- `--skip-training`: Skip the model training step
- `--skip-business-analysis`: Skip the business impact analysis step
- `--launch-dashboard`: Launch the interactive dashboard after completion

Example:
```bash
python run_analysis.py --skip-preprocessing --launch-dashboard
```

### Running Individual Components

You can also run individual components of the pipeline separately:

**Data Acquisition**:
```bash
python -c "from src.data.acquisition import main; main()"
```

**Data Preprocessing**:
```bash
python -c "from src.data.preprocessing import main; main()"
```

**Model Training**:
```bash
python -c "from src.models.train_model import main; main()"
```

**Business Analysis**:
```bash
python -c "from src.analysis.business_analysis import main; main()"
```

### Running the Interactive Dashboard

To launch the interactive dashboard:

```bash
python dashboard/app.py
```

Then open your web browser and navigate to `http://localhost:8050`

### Exploring via Jupyter Notebooks

For interactive exploration of the data and analysis, you can use the Jupyter notebooks:

```bash
jupyter notebook notebooks/1_exploratory_analysis.ipynb
```

## Customization

### Modifying Model Parameters

To modify the model parameters, edit the model creation functions in `src/models/train_model.py`. For example, to change the XGBoost hyperparameters:

```python
def train_xgboost(X_train, y_train, scale_pos_weight=None):
    # Modify parameters here
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,  # Changed from 100
        learning_rate=0.05,  # Changed from 0.1
        max_depth=6,  # Changed from 5
        # ... other parameters
    )
    
```

### Adding New Features

To add new features to the analysis, modify the `create_features` function in `src/data/preprocessing.py`.

### Customizing Business Analysis

To adjust the business impact analysis parameters (e.g., readmission costs, intervention effectiveness), modify the parameters in `src/analysis/business_analysis.py`.

## Troubleshooting

### Common Issues

**Package Installation Errors**:
- If you encounter issues with XGBoost or LightGBM installation, try installing them separately:
  ```bash
  pip install xgboost==1.6.1
  pip install lightgbm==3.3.2
  ```

**Memory Issues**:
- If you encounter memory errors during model training, try reducing the complexity of the models by modifying parameters in `src/models/train_model.py`.

**Dataset Download Issues**:
- If automatic dataset download fails, manually download the dataset from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/diabetes+130-us+hospitals+for+years+1999-2008) and place the files in `data/raw/dataset_diabetes/`.

### Getting Help

If you encounter any issues or have questions about the project:
1. Check the generated log file (`project.log`) for error messages
2. Refer to the docstrings in the source code for function documentation
