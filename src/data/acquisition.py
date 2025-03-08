"""
Data Acquisition Script for Hospital Readmission Project
Downloads and processes the Diabetes 130-US Hospitals dataset
"""

import os
import urllib.request
import zipfile
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# URLs and file paths
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip"
DATA_DIR = Path("./data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

def create_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Created directory: {dir_path}")

def download_data():
    """Download the dataset from UCI repository"""
    zip_path = RAW_DATA_DIR / "dataset_diabetes.zip"
    
    if zip_path.exists():
        logger.info(f"Dataset already downloaded to {zip_path}")
    else:
        logger.info(f"Downloading dataset from {DATA_URL}")
        try:
            urllib.request.urlretrieve(DATA_URL, zip_path)
            logger.info(f"Dataset downloaded to {zip_path}")
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise
    
    return zip_path

def extract_data(zip_path):
    """Extract the downloaded zip file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        logger.info(f"Dataset extracted to {RAW_DATA_DIR}")
    except Exception as e:
        logger.error(f"Failed to extract dataset: {e}")
        raise

def load_and_verify_data():
    """Load the dataset and verify its contents"""
    data_path = RAW_DATA_DIR / "dataset_diabetes/diabetic_data.csv"
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Dataset loaded with shape: {df.shape}")
        
        # Basic verification
        expected_columns = ['race', 'gender', 'age', 'admission_type_id', 
                           'discharge_disposition_id', 'admission_source_id',
                           'time_in_hospital', 'medical_specialty', 
                           'num_lab_procedures', 'num_procedures', 
                           'num_medications', 'number_outpatient', 
                           'number_emergency', 'number_inpatient', 
                           'diag_1', 'diag_2', 'diag_3', 'number_diagnoses', 
                           'max_glu_serum', 'A1Cresult', 'metformin', 
                           'repaglinide', 'nateglinide', 'chlorpropamide', 
                           'glimepiride', 'acetohexamide', 'glipizide', 
                           'glyburide', 'tolbutamide', 'pioglitazone', 
                           'rosiglitazone', 'acarbose', 'miglitol', 
                           'troglitazone', 'tolazamide', 'examide', 
                           'citoglipton', 'insulin', 'glyburide-metformin', 
                           'glipizide-metformin', 'glimepiride-pioglitazone', 
                           'metformin-rosiglitazone', 'metformin-pioglitazone', 
                           'change', 'diabetesMed', 'readmitted']
        
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing expected columns: {missing_columns}")
        
        # Check target variable
        if 'readmitted' in df.columns:
            readmission_counts = df['readmitted'].value_counts()
            logger.info(f"Readmission distribution: {readmission_counts}")
        
        # Save a copy to processed directory
        processed_path = PROCESSED_DATA_DIR / "diabetes_hospital_data.csv"
        df.to_csv(processed_path, index=False)
        logger.info(f"Raw data saved to {processed_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Failed to load and verify dataset: {e}")
        raise

def get_IDs_mapping():
    """Load the IDs_mapping.csv file or create it if needed"""
    mapping_path = RAW_DATA_DIR / "dataset_diabetes/IDs_mapping.csv"
    
    if not mapping_path.exists():
        logger.warning(f"IDs mapping file not found at {mapping_path}")
        # Create a simplified mapping file for demo purposes
        admission_type_mapping = {
            1: "Emergency",
            2: "Urgent",
            3: "Elective",
            4: "Newborn",
            5: "Not Available",
            6: "NULL",
            7: "Trauma Center",
            8: "Not Mapped"
        }
        
        discharge_disposition_mapping = {
            1: "Discharged to home",
            2: "Discharged/transferred to another short term hospital",
            3: "Discharged/transferred to SNF",
            4: "Discharged/transferred to ICF",
            5: "Discharged/transferred to another type of inpatient care institution",
            6: "Discharged/transferred to home with home health service",
            7: "Left AMA",
            8: "Discharged/transferred to home under care of Home IV provider",
            9: "Admitted as an inpatient to this hospital",
            10: "Neonate discharged to another hospital for neonatal aftercare",
            11: "Expired",
            12: "Still patient or expected to return for outpatient services",
            13: "Hospice / home",
            14: "Hospice / medical facility",
            15: "Discharged/transferred within this institution to Medicare approved swing bed",
            16: "Discharged/transferred/referred another institution for outpatient services",
            17: "Discharged/transferred/referred to this institution for outpatient services",
            18: "NULL",
            19: "Expired at home. Medicaid only, hospice.",
            20: "Expired in a medical facility. Medicaid only, hospice.",
            21: "Expired, place unknown. Medicaid only, hospice.",
            22: "Discharged/transferred to another rehab fac including rehab units of a hospital.",
            23: "Discharged/transferred to a long term care hospital.",
            24: "Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.",
            25: "Not Mapped",
            26: "Unknown/Invalid",
            30: "Discharged/transferred to another Type of Health Care Institution not Defined Elsewhere",
            27: "Discharged/transferred to a federal health care facility.",
            28: "Discharged/transferred to a psychiatric hospital/unit of hospital",
            29: "Discharged/transferred to a Critical Access Hospital (CAH)."
        }
        
        admission_source_mapping = {
            1: "Physician Referral",
            2: "Clinic Referral",
            3: "HMO Referral",
            4: "Transfer from a hospital",
            5: "Transfer from a Skilled Nursing Facility (SNF)",
            6: "Transfer from another health care facility",
            7: "Emergency Room",
            8: "Court/Law Enforcement",
            9: "Not Available",
            10: "Transfer from critial access hospital",
            11: "Normal Delivery",
            12: "Premature Delivery",
            13: "Sick Baby",
            14: "Extramural Birth",
            15: "Not Available",
            17: "NULL",
            18: "Transfer From Another Home Health Agency",
            19: "Readmission to Same Home Health Agency",
            20: "Not Mapped",
            21: "Unknown/Invalid",
            22: "Transfer from hospital inpt/same fac reslt in a sep claim",
            23: "Born inside this hospital",
            24: "Born outside this hospital",
            25: "Transfer from Ambulatory Surgery Center",
            26: "Transfer from Hospice"
        }
        
        # Create DataFrames
        admission_type_df = pd.DataFrame(list(admission_type_mapping.items()), 
                                        columns=['id', 'description'])
        admission_type_df['category'] = 'admission_type'
        
        discharge_disposition_df = pd.DataFrame(list(discharge_disposition_mapping.items()), 
                                              columns=['id', 'description'])
        discharge_disposition_df['category'] = 'discharge_disposition'
        
        admission_source_df = pd.DataFrame(list(admission_source_mapping.items()), 
                                         columns=['id', 'description'])
        admission_source_df['category'] = 'admission_source'
        
        # Combine into one DataFrame
        mappings_df = pd.concat([admission_type_df, 
                                discharge_disposition_df, 
                                admission_source_df])
        
        # Save to CSV
        mappings_df.to_csv(PROCESSED_DATA_DIR / "IDs_mapping.csv", index=False)
        logger.info(f"Created IDs mapping file at {PROCESSED_DATA_DIR / 'IDs_mapping.csv'}")
        
        return mappings_df
    
    try:
        mappings_df = pd.read_csv(mapping_path)
        processed_mapping_path = PROCESSED_DATA_DIR / "IDs_mapping.csv"
        mappings_df.to_csv(processed_mapping_path, index=False)
        logger.info(f"IDs mapping loaded and saved to {processed_mapping_path}")
        return mappings_df
    
    except Exception as e:
        logger.error(f"Failed to load IDs mapping: {e}")
        raise

def main():
    """Main function to orchestrate data acquisition"""
    try:
        create_directories()
        zip_path = download_data()
        extract_data(zip_path)
        df = load_and_verify_data()
        mappings = get_IDs_mapping()
        
        logger.info("Data acquisition completed successfully")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset columns: {df.columns.tolist()}")
        
        return df, mappings
    
    except Exception as e:
        logger.error(f"Data acquisition failed: {e}")
        raise

if __name__ == "__main__":
    main()
