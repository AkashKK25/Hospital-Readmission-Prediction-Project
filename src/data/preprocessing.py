"""
Preprocessing module for the Hospital Readmission Project
Handles data cleaning, feature engineering, and preparation for modeling
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import re
import sklearn
from packaging import version

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path("./data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_PATH = PROCESSED_DATA_DIR / "diabetes_hospital_data.csv"
MAPPINGS_PATH = PROCESSED_DATA_DIR / "IDs_mapping.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "diabetes_processed.csv"

def load_raw_data():
    """Load the raw diabetes dataset"""
    try:
        df = pd.read_csv(RAW_DATA_PATH)
        logger.info(f"Raw data loaded with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        raise

def load_mappings():
    """Load the ID mappings"""
    try:
        mappings = pd.read_csv(MAPPINGS_PATH)
        logger.info(f"ID mappings loaded with shape: {mappings.shape}")
        
        # Convert to usable dictionaries
        mapping_dict = {}
        
        # Check if 'category' column exists
        if 'category' not in mappings.columns:
            # Create simpler mappings without categories
            if 'id' in mappings.columns and 'description' in mappings.columns:
                # For admission_type
                admission_rows = mappings[mappings['description'].str.contains('admission type', case=False, na=False)]
                if not admission_rows.empty:
                    mapping_dict['admission_type'] = dict(zip(admission_rows['id'], admission_rows['description']))
                
                # For discharge_disposition
                discharge_rows = mappings[mappings['description'].str.contains('discharge', case=False, na=False)]
                if not discharge_rows.empty:
                    mapping_dict['discharge_disposition'] = dict(zip(discharge_rows['id'], discharge_rows['description']))
                
                # For admission_source
                source_rows = mappings[mappings['description'].str.contains('admission source', case=False, na=False)]
                if not source_rows.empty:
                    mapping_dict['admission_source'] = dict(zip(source_rows['id'], source_rows['description']))
            else:
                # Create dummy mapping dictionaries
                logger.warning("Creating default mapping dictionaries as 'category' column is missing")
                mapping_dict['admission_type'] = {1: "Emergency", 2: "Urgent", 3: "Elective", 4: "Newborn"}
                mapping_dict['discharge_disposition'] = {1: "Discharged to home", 2: "Transferred to short term hospital", 11: "Expired"}
                mapping_dict['admission_source'] = {1: "Physician Referral", 7: "Emergency Room", 2: "Clinic Referral"}
        else:
            # Original code path when 'category' exists
            for category in mappings['category'].unique():
                category_df = mappings[mappings['category'] == category]
                mapping_dict[category] = dict(zip(category_df['id'], category_df['description']))
        
        return mapping_dict
    except Exception as e:
        logger.error(f"Failed to load ID mappings: {e}")
        raise

def clean_data(df, mappings):
    """
    Clean the dataset by:
    - Handling missing values
    - Decoding ID fields
    - Recoding categorical variables
    - Fixing data types
    """
    try:
        df_clean = df.copy()
        
        # Replace '?' with NaN
        df_clean.replace('?', np.nan, inplace=True)
        
        # Check for missing values
        missing_values = df_clean.isnull().sum()
        logger.info(f"Missing values before imputation:\n{missing_values[missing_values > 0]}")
        
        # Map ID fields to their descriptions
        if 'admission_type_id' in df_clean.columns and 'admission_type' in mappings:
            df_clean['admission_type'] = df_clean['admission_type_id'].map(mappings['admission_type'])
            
        if 'discharge_disposition_id' in df_clean.columns and 'discharge_disposition' in mappings:
            df_clean['discharge_disposition'] = df_clean['discharge_disposition_id'].map(mappings['discharge_disposition'])
            
        if 'admission_source_id' in df_clean.columns and 'admission_source' in mappings:
            df_clean['admission_source'] = df_clean['admission_source_id'].map(mappings['admission_source'])
        
        # Convert numeric columns to appropriate types
        numeric_cols = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses'
        ]
        
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        # Process medication columns - most have 'No', 'Up', 'Down', 'Steady'
        medication_cols = [
            'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
            'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
            'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
            'miglitol', 'troglitazone', 'tolazamide', 'examide',
            'citoglipton', 'insulin', 'glyburide-metformin',
            'glipizide-metformin', 'glimepiride-pioglitazone',
            'metformin-rosiglitazone', 'metformin-pioglitazone'
        ]
        
        for col in medication_cols:
            if col in df_clean.columns:
                # Create a binary indicator for medication use (any dosage)
                df_clean[f'{col}_used'] = df_clean[col].apply(lambda x: 0 if x == 'No' else 1)
                
                # Create a numeric dosage change indicator
                dosage_map = {'No': 0, 'Steady': 1, 'Up': 2, 'Down': -1}
                df_clean[f'{col}_dosage'] = df_clean[col].map(dosage_map)
        
        # Process diagnosis codes (ICD-9)
        diag_cols = ['diag_1', 'diag_2', 'diag_3']
        
        # Define diagnosis categories based on ICD-9 ranges
        def categorize_diagnosis(code):
            try:
                if pd.isna(code) or code == '?':
                    return 'Unknown'
                    
                # Convert to string and clean
                code_str = str(code).strip()
                
                # Check if it's the ICD-9 code format with V or E prefix
                if code_str.startswith('V'):
                    return 'V_codes'  # Supplementary classification
                elif code_str.startswith('E'):
                    return 'E_codes'  # External causes
                    
                # Try to convert to numeric for range checking
                code_num = float(code_str)
                
                # Categorize based on ICD-9 chapter ranges
                if 1 <= code_num < 140:
                    return 'Infectious'
                elif 140 <= code_num < 240:
                    return 'Neoplasms'
                elif 240 <= code_num < 280:
                    return 'Endocrine'  # Includes diabetes
                elif 280 <= code_num < 290:
                    return 'Blood'
                elif 290 <= code_num < 320:
                    return 'Mental'
                elif 320 <= code_num < 390:
                    return 'Nervous'
                elif 390 <= code_num < 460:
                    return 'Circulatory'  # Heart disease, hypertension
                elif 460 <= code_num < 520:
                    return 'Respiratory'
                elif 520 <= code_num < 580:
                    return 'Digestive'
                elif 580 <= code_num < 630:
                    return 'Genitourinary'
                elif 630 <= code_num < 680:
                    return 'Pregnancy'
                elif 680 <= code_num < 710:
                    return 'Skin'
                elif 710 <= code_num < 740:
                    return 'Musculoskeletal'
                elif 740 <= code_num < 760:
                    return 'Congenital'
                elif 760 <= code_num < 780:
                    return 'Perinatal'
                elif 780 <= code_num < 800:
                    return 'Symptoms'
                elif 800 <= code_num < 1000:
                    return 'Injury'
                else:
                    return 'Other'
            except:
                return 'Unknown'
        
        # Apply categorization to diagnosis columns
        for col in diag_cols:
            if col in df_clean.columns:
                df_clean[f'{col}_category'] = df_clean[col].apply(categorize_diagnosis)
                
                # Create diabetes-specific indicator
                is_diabetes = df_clean[col].apply(lambda x: 
                                                 True if not pd.isna(x) and str(x).startswith('250') 
                                                 else False)
                df_clean[f'{col}_is_diabetes'] = is_diabetes.astype(int)
                
                # Create circulatory system indicator (heart disease, hypertension)
                is_circulatory = df_clean[f'{col}_category'] == 'Circulatory'
                df_clean[f'{col}_is_circulatory'] = is_circulatory.astype(int)
                
                # Create respiratory system indicator
                is_respiratory = df_clean[f'{col}_category'] == 'Respiratory'
                df_clean[f'{col}_is_respiratory'] = is_respiratory.astype(int)
        
        # Process A1C and glucose results
        if 'A1Cresult' in df_clean.columns:
            a1c_map = {'>8': 3, '>7': 2, 'Norm': 1, 'None': 0}
            df_clean['A1C_value'] = df_clean['A1Cresult'].map(a1c_map)
            
        if 'max_glu_serum' in df_clean.columns:
            glu_map = {'>300': 3, '>200': 2, 'Norm': 1, 'None': 0}
            df_clean['glucose_value'] = df_clean['max_glu_serum'].map(glu_map)
        
        # Process the target variable 'readmitted'
        if 'readmitted' in df_clean.columns:
            # Create binary target: readmitted within 30 days (1) vs. not (0)
            df_clean['readmitted_30d'] = df_clean['readmitted'].apply(
                lambda x: 1 if x == '<30' else 0
            )
            
            # Also create a 3-class version: <30 days, >30 days, No readmission
            readmit_map = {'<30': 2, '>30': 1, 'NO': 0}
            df_clean['readmitted_class'] = df_clean['readmitted'].map(readmit_map)
        
        # Drop the original ID columns if we've created the mapped versions
        if 'admission_type' in df_clean.columns:
            df_clean.drop('admission_type_id', axis=1, inplace=True, errors='ignore')
            
        if 'discharge_disposition' in df_clean.columns:
            df_clean.drop('discharge_disposition_id', axis=1, inplace=True, errors='ignore')
            
        if 'admission_source' in df_clean.columns:
            df_clean.drop('admission_source_id', axis=1, inplace=True, errors='ignore')
        
        # Check for missing values after processing
        missing_after = df_clean.isnull().sum()
        logger.info(f"Missing values after processing:\n{missing_after[missing_after > 0]}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Failed during data cleaning: {e}")
        raise

def create_features(df):
    """Create additional features for modeling"""
    try:
        df_featured = df.copy()
        
        # Age category feature
        if 'age' in df_featured.columns:
            # Create numeric age from categorical ranges
            def age_to_numeric(age_range):
                age_map = {
                    '[0-10)': 5,
                    '[10-20)': 15,
                    '[20-30)': 25,
                    '[30-40)': 35,
                    '[40-50)': 45,
                    '[50-60)': 55,
                    '[60-70)': 65,
                    '[70-80)': 75,
                    '[80-90)': 85,
                    '[90-100)': 95
                }
                return age_map.get(age_range, 50)  # Default to 50 if unknown
                
            df_featured['age_numeric'] = df_featured['age'].apply(age_to_numeric)
            
            # Create age group categories
            df_featured['age_group'] = pd.cut(
                df_featured['age_numeric'],
                bins=[0, 30, 50, 70, 100],
                labels=['Young', 'Middle', 'Senior', 'Elderly']
            )
        
        # Total number of visits feature
        visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
        if all(col in df_featured.columns for col in visit_cols):
            df_featured['total_visits'] = df_featured[visit_cols].sum(axis=1)
        
        # Medication complexity features
        med_used_cols = [col for col in df_featured.columns if col.endswith('_used')]
        if med_used_cols:
            # Total number of medications used
            df_featured['total_meds_used'] = df_featured[med_used_cols].sum(axis=1)
            
            # Medication diversity ratio (unique meds / total possible meds)
            df_featured['med_diversity_ratio'] = df_featured['total_meds_used'] / len(med_used_cols)
            
        # Diagnosis complexity feature
        if 'number_diagnoses' in df_featured.columns:
            # Create categories for diagnosis count
            df_featured['diagnosis_complexity'] = pd.cut(
                df_featured['number_diagnoses'],
                bins=[0, 3, 6, 9, 20],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Hospital stay features
        if 'time_in_hospital' in df_featured.columns:
            # Create categories for length of stay
            df_featured['stay_length_cat'] = pd.cut(
                df_featured['time_in_hospital'],
                bins=[0, 3, 7, 14, 100],
                labels=['Short', 'Medium', 'Long', 'Very Long']
            )
            
        # Lab procedure intensity
        if 'num_lab_procedures' in df_featured.columns:
            df_featured['lab_intensity'] = pd.cut(
                df_featured['num_lab_procedures'],
                bins=[0, 25, 50, 75, 1000],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
        
        # Feature for patients with diabetes as primary diagnosis
        diag_diabetes_cols = [col for col in df_featured.columns if col.endswith('_is_diabetes')]
        if diag_diabetes_cols and len(diag_diabetes_cols) >= 1:
            df_featured['primary_diabetes'] = df_featured['diag_1_is_diabetes']
            
            if len(diag_diabetes_cols) >= 3:
                # Any diabetes diagnosis
                df_featured['any_diabetes_diag'] = (
                    df_featured['diag_1_is_diabetes'] | 
                    df_featured['diag_2_is_diabetes'] | 
                    df_featured['diag_3_is_diabetes']
                ).astype(int)
        
        # Comorbidity features
        # Check for common comorbidities (circulatory, respiratory)
        circ_cols = [col for col in df_featured.columns if col.endswith('_is_circulatory')]
        resp_cols = [col for col in df_featured.columns if col.endswith('_is_respiratory')]
        
        if circ_cols:
            df_featured['has_circulatory_disease'] = df_featured[circ_cols].max(axis=1)
            
        if resp_cols:
            df_featured['has_respiratory_disease'] = df_featured[resp_cols].max(axis=1)
            
        if circ_cols and resp_cols:
            df_featured['circulatory_respiratory_comorbidity'] = (
                (df_featured['has_circulatory_disease'] == 1) & 
                (df_featured['has_respiratory_disease'] == 1)
            ).astype(int)
        
        # Insulin-related features
        if 'insulin_used' in df_featured.columns:
            # Combine insulin usage with A1C levels for a risk feature
            if 'A1C_value' in df_featured.columns:
                df_featured['insulin_with_high_A1C'] = (
                    (df_featured['insulin_used'] == 1) & 
                    (df_featured['A1C_value'] >= 2)  # A1C > 7
                ).astype(int)
        
        # Multiple medication changes feature
        dosage_cols = [col for col in df_featured.columns if col.endswith('_dosage')]
        if dosage_cols:
            # Count medication changes (up or down)
            df_featured['med_changes_count'] = df_featured[dosage_cols].apply(
                lambda row: sum([1 for val in row if val != 0 and val != 1]), axis=1
            )
        
        logger.info(f"Feature engineering complete. New shape: {df_featured.shape}")
        return df_featured
        
    except Exception as e:
        logger.error(f"Failed during feature creation: {e}")
        raise

def prepare_modeling_data(df, target_col='readmitted_30d', test_size=0.2, random_state=42):
    """
    Prepare the final dataset for modeling:
    - Select relevant features
    - Handle remaining missing values
    - Convert categorical variables
    - Scale numerical features
    - Split into train/test sets
    """
    try:
        from sklearn.model_selection import train_test_split
        
        df_model = df.copy()
        
        # Ensure target column exists
        if target_col not in df_model.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataset")
        
        # Define feature groups
        
        # Important categorical features to encode
        cat_features = [
            'gender', 'race', 'age', 'admission_type', 'discharge_disposition',
            'admission_source', 'medical_specialty', 'diag_1_category',
            'diag_2_category', 'diag_3_category', 'A1Cresult', 'max_glu_serum',
            'age_group', 'diagnosis_complexity', 'stay_length_cat', 'lab_intensity'
        ]
        
        # Important numerical features to scale
        num_features = [
            'time_in_hospital', 'num_lab_procedures', 'num_procedures',
            'num_medications', 'number_outpatient', 'number_emergency',
            'number_inpatient', 'number_diagnoses', 'age_numeric',
            'total_visits', 'total_meds_used', 'med_diversity_ratio',
            'med_changes_count'
        ]
        
        # Binary features (already encoded)
        binary_features = [
            'diabetesMed', 'change', 'primary_diabetes', 'any_diabetes_diag',
            'has_circulatory_disease', 'has_respiratory_disease',
            'circulatory_respiratory_comorbidity', 'insulin_with_high_A1C'
        ]
        
        # Binary medication indicators
        med_features = [col for col in df_model.columns if col.endswith('_used')]
        
        # Filter to include only features that exist in the dataframe
        cat_features = [col for col in cat_features if col in df_model.columns]
        num_features = [col for col in num_features if col in df_model.columns]
        binary_features = [col for col in binary_features if col in df_model.columns]
        med_features = [col for col in med_features if col in df_model.columns]
        
        # Check for duplicate column names in each feature group
        all_features = []
        for feature_list in [cat_features, num_features, binary_features, med_features]:
            # Add only unique columns not already in all_features
            for feature in feature_list:
                if feature not in all_features:
                    all_features.append(feature)
        
        logger.info(f"Selected {len(all_features)} features for modeling")
        
        # Create a new dataframe with only the selected features
        X = df_model[all_features].copy()
        y = df_model[target_col].copy()
        
        # Double check for duplicate columns in X
        duplicate_cols = X.columns[X.columns.duplicated()]
        if len(duplicate_cols) > 0:
            logger.warning(f"Found duplicate columns: {duplicate_cols.tolist()}")
            # Create a new dataframe with unique column names
            X_columns = list(X.columns)
            unique_columns = []
            for i, col in enumerate(X_columns):
                if col in unique_columns:
                    new_col = f"{col}_{unique_columns.count(col)}"
                    logger.info(f"Renaming duplicate column {col} to {new_col}")
                    X_columns[i] = new_col
                unique_columns.append(X_columns[i])
            X.columns = X_columns
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Determine which columns are numerical and categorical after potential renaming
        num_features = [col for col in X.columns if col in num_features or any(col.startswith(f) and col.endswith('_0') for f in num_features)]
        cat_features = [col for col in X.columns if col in cat_features or any(col.startswith(f) and col.endswith('_0') for f in cat_features)]
        
        # Create preprocessing pipeline
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.compose import ColumnTransformer
        
        # Create feature transformation pipelines
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Use sparse_output instead of sparse for newer sklearn versions
        import sklearn
        from packaging import version
        
        if version.parse(sklearn.__version__) >= version.parse('1.2.0'):
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
            ])
        
        # Check for any remaining string/object columns that might have been missed
        object_columns = X_train.select_dtypes(include=['object']).columns.tolist()
        if object_columns:
            logger.warning(f"Found {len(object_columns)} object columns that need encoding: {object_columns}")
            # Add these to cat_features if not already included
            for col in object_columns:
                if col not in cat_features:
                    cat_features.append(col)
                    logger.info(f"Added {col} to categorical features list")
        
        # Apply different preprocessing to different feature types
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_features),
                ('cat', categorical_transformer, cat_features)
            ],
            remainder='passthrough'  # Keep binary features as is
        )
        
        # Fit the preprocessor to the training data
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        
        # Get feature names for categorical variables after one-hot encoding
        cat_feature_names = []
        if len(cat_features) > 0:  # Only do this if we have categorical features
            for i, feature in enumerate(cat_features):
                encoder = preprocessor.transformers_[1][1].named_steps['onehot']
                feature_categories = encoder.categories_[i]
                cat_feature_names.extend([f"{feature}_{cat}" for cat in feature_categories])
        
        # Get remaining feature names
        remaining_features = [col for col in X.columns if col not in num_features and col not in cat_features]
        
        # Combine all feature names
        feature_names = num_features + cat_feature_names + remaining_features
        
        # Save the model-ready data
        processed_data = {
            'X_train': X_train_processed,
            'X_test': X_test_processed,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_names,
            'preprocessor': preprocessor
        }
        
        # Save preprocessed data for modeling
        import joblib
        joblib.dump(processed_data, PROCESSED_DATA_DIR / "model_ready_data.pkl")
        logger.info(f"Model-ready data saved to {PROCESSED_DATA_DIR / 'model_ready_data.pkl'}")
        
        # Also save the pre-processed dataframe for reference
        df_model.to_csv(PROCESSED_DATA_PATH, index=False)
        logger.info(f"Processed dataframe saved to {PROCESSED_DATA_PATH}")
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed during modeling data preparation: {e}")
        raise


def main():
    """Main function to orchestrate data preprocessing"""
    try:
        # Load data
        df = load_raw_data()
        mappings = load_mappings()
        
        # Clean data
        df_clean = clean_data(df, mappings)
        logger.info("Data cleaning complete")
        
        # Create features
        df_featured = create_features(df_clean)
        logger.info("Feature engineering complete")
        
        # Prepare for modeling
        processed_data = prepare_modeling_data(df_featured)
        logger.info("Data preprocessing pipeline complete")
        
        return df_featured, processed_data
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()
