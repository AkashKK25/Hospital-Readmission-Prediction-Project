"""
Model Training Module for Hospital Readmission Prediction
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# File paths
DATA_DIR = Path("./data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = Path("./models")
RESULTS_DIR = Path("./reports/results")
FIGURES_DIR = Path("./reports/figures")

def create_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            logger.info(f"Created directory: {dir_path}")

def load_processed_data():
    """Load the preprocessed data for modeling"""
    try:
        data_path = PROCESSED_DATA_DIR / "model_ready_data.pkl"
        processed_data = joblib.load(data_path)
        logger.info(f"Loaded processed data from {data_path}")
        
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        
        # Check for string values in the data
        if isinstance(X_train, np.ndarray):
            # Check if any element is a string
            if X_train.dtype == object:
                logger.warning("Found object dtype in X_train, converting strings to numeric...")
                # Try to convert strings to numeric (one-hot encode any remaining strings)
                from sklearn.preprocessing import OneHotEncoder
                import pandas as pd
                
                # Convert to DataFrame for easier handling
                X_train_df = pd.DataFrame(X_train)
                X_test_df = pd.DataFrame(X_test)
                
                # Find columns with string values
                string_cols = []
                for col in X_train_df.columns:
                    if X_train_df[col].dtype == object:
                        try:
                            # Try to convert to numeric
                            X_train_df[col] = pd.to_numeric(X_train_df[col], errors='raise')
                            X_test_df[col] = pd.to_numeric(X_test_df[col], errors='raise')
                        except:
                            string_cols.append(col)
                
                if string_cols:
                    logger.warning(f"Found string columns: {string_cols}")
                    # One-hot encode string columns
                    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    
                    # Get string column data
                    X_train_str = X_train_df[string_cols].values
                    X_test_str = X_test_df[string_cols].values
                    
                    # Fit and transform
                    X_train_str_encoded = ohe.fit_transform(X_train_str)
                    X_test_str_encoded = ohe.transform(X_test_str)
                    
                    # Get non-string columns
                    non_str_cols = [c for c in X_train_df.columns if c not in string_cols]
                    X_train_non_str = X_train_df[non_str_cols].values
                    X_test_non_str = X_test_df[non_str_cols].values
                    
                    # Combine encoded and non-encoded parts
                    X_train = np.hstack([X_train_non_str, X_train_str_encoded])
                    X_test = np.hstack([X_test_non_str, X_test_str_encoded])
                    
                    logger.info(f"Successfully encoded string columns, new shape: {X_train.shape}")
                else:
                    # If no string columns were found, convert back to numpy array
                    X_train = X_train_df.values
                    X_test = X_test_df.values
        
        # Update processed data with fixed arrays
        processed_data['X_train'] = X_train
        processed_data['X_test'] = X_test
        
        logger.info(f"Loaded data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
        logger.info(f"Class distribution in training set: {np.bincount(y_train)}")
        logger.info(f"Class distribution in test set: {np.bincount(y_test)}")
        
        return processed_data
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        raise
    
def train_logistic_regression(X_train, y_train, class_weight='balanced'):
    """Train a logistic regression model"""
    try:
        # Create and train the model
        lr_model = LogisticRegression(
            C=1.0,
            class_weight=class_weight,
            solver='liblinear',
            random_state=42,
            max_iter=1000
        )
        
        lr_model.fit(X_train, y_train)
        logger.info("Logistic Regression model trained successfully")
        
        return lr_model
    except Exception as e:
        logger.error(f"Failed to train Logistic Regression model: {e}")
        raise

def train_random_forest(X_train, y_train, class_weight='balanced'):
    """Train a random forest model"""
    try:
        # Create and train the model
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=4,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        logger.info("Random Forest model trained successfully")
        
        return rf_model
    except Exception as e:
        logger.error(f"Failed to train Random Forest model: {e}")
        raise

def train_gradient_boosting(X_train, y_train):
    """Train a gradient boosting model"""
    try:
        # Create and train the model
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        gb_model.fit(X_train, y_train)
        logger.info("Gradient Boosting model trained successfully")
        
        return gb_model
    except Exception as e:
        logger.error(f"Failed to train Gradient Boosting model: {e}")
        raise

def train_xgboost(X_train, y_train, scale_pos_weight=None):
    """Train an XGBoost model"""
    try:
        # Create and train the model
        if scale_pos_weight is None:
            # Calculate class weight based on class distribution
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        xgb_model.fit(X_train, y_train)
        logger.info("XGBoost model trained successfully")
        
        return xgb_model
    except Exception as e:
        logger.error(f"Failed to train XGBoost model: {e}")
        raise

def train_lightgbm(X_train, y_train, class_weight=None):
    """Train a LightGBM model"""
    try:
        # Create and train the model
        if class_weight is None:
            # Calculate class weight based on class distribution
            neg_count = np.sum(y_train == 0)
            pos_count = np.sum(y_train == 1)
            class_weight = {0: 1.0, 1: neg_count / pos_count if pos_count > 0 else 1.0}
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight=class_weight,
            random_state=42
        )
        
        lgb_model.fit(X_train, y_train)
        logger.info("LightGBM model trained successfully")
        
        return lgb_model
    except Exception as e:
        logger.error(f"Failed to train LightGBM model: {e}")
        raise

def train_with_smote(model, X_train, y_train, random_state=42):
    """Train a model with SMOTE oversampling for imbalanced data"""
    try:
        # Create a pipeline with SMOTE and the classifier
        smote_pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('classifier', model)
        ])
        
        # Fit the pipeline
        smote_pipeline.fit(X_train, y_train)
        logger.info(f"Model with SMOTE trained successfully: {type(model).__name__}")
        
        return smote_pipeline
    except Exception as e:
        logger.error(f"Failed to train model with SMOTE: {e}")
        raise

def evaluate_model(model, X_test, y_test, model_name, threshold=0.5):
    """Evaluate a trained model and return metrics"""
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get probability predictions for AUC and PR curves if possible
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'predict_proba'):
            y_prob = model.named_steps['classifier'].predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred  # Fall back to binary predictions if probabilities not available
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # AUC and average precision
        if len(np.unique(y_test)) > 1:  # Only calculate if both classes present
            roc_auc = roc_auc_score(y_test, y_prob)
            avg_precision = average_precision_score(y_test, y_prob)
        else:
            roc_auc = np.nan
            avg_precision = np.nan
        
        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Collect results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_true': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob
        }
        
        logger.info(f"Evaluation results for {model_name}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    except Exception as e:
        logger.error(f"Failed to evaluate model {model_name}: {e}")
        raise

def plot_roc_curve(results_dict, save_path=None):
    """Plot ROC curves for all models"""
    try:
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = results['roc_auc']
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot the random baseline
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot ROC curves: {e}")
        raise

def plot_precision_recall_curve(results_dict, save_path=None):
    """Plot Precision-Recall curves for all models"""
    try:
        plt.figure(figsize=(10, 8))
        
        for model_name, results in results_dict.items():
            y_true = results['y_true']
            y_prob = results['y_prob']
            
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            avg_precision = results['avg_precision']
            
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
        
        # Calculate the no-skill baseline (prevalence)
        prevalence = np.sum(y_true) / len(y_true) if len(y_true) > 0 else 0
        plt.plot([0, 1], [prevalence, prevalence], 'k--', lw=2, label=f'No Skill (AP = {prevalence:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve plot saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot Precision-Recall curves: {e}")
        raise

def plot_confusion_matrices(results_dict, save_path=None):
    """Plot confusion matrices for all models"""
    try:
        n_models = len(results_dict)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(results_dict.items()):
            cm = results['confusion_matrix']
            
            # Create a prettier confusion matrix with percentages
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_norm = np.nan_to_num(cm_norm)  # Replace NaN with 0
            
            # Plot the confusion matrix
            sns.heatmap(
                cm, 
                annot=True, 
                fmt="d", 
                cmap="Blues", 
                ax=axes[i],
                cbar=False
            )
            
            # Add percentage labels
            for j in range(cm.shape[0]):
                for k in range(cm.shape[1]):
                    axes[i].text(
                        k + 0.5, 
                        j + 0.5, 
                        f"{cm_norm[j, k]:.1%}",
                        ha="center", 
                        va="center", 
                        fontsize=9,
                        color="white" if cm_norm[j, k] > 0.5 else "black"
                    )
            
            axes[i].set_title(f"{model_name}")
            axes[i].set_xlabel("Predicted label")
            axes[i].set_ylabel("True label")
            axes[i].set_xticklabels(['Not Readmitted', 'Readmitted'])
            axes[i].set_yticklabels(['Not Readmitted', 'Readmitted'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrices plot saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot confusion matrices: {e}")
        raise

def plot_feature_importance(model, feature_names, model_name, top_n=20, save_path=None):
    """Plot feature importance for the model"""
    try:
        plt.figure(figsize=(12, 10))
        
        if hasattr(model, 'feature_importances_'):
            # For tree-based models that have feature_importances_ attribute
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            # For linear models like logistic regression
            importances = np.abs(model.coef_[0])
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'feature_importances_'):
            # For pipeline with classifier that has feature_importances_
            importances = model.named_steps['classifier'].feature_importances_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'coef_'):
            # For pipeline with classifier that has coef_
            importances = np.abs(model.named_steps['classifier'].coef_[0])
        else:
            logger.warning(f"Model {model_name} does not have feature importances or coefficients")
            return
        
        # Get feature importance and names
        if len(importances) > len(feature_names):
            logger.warning(f"Mismatch between importances length {len(importances)} and feature names length {len(feature_names)}")
            # Truncate importances to match feature_names
            importances = importances[:len(feature_names)]
        elif len(importances) < len(feature_names):
            logger.warning(f"Mismatch between importances length {len(importances)} and feature names length {len(feature_names)}")
            # Truncate feature_names to match importances
            feature_names = feature_names[:len(importances)]
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance and get top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
        
        # Plot horizontal bar chart
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f"Top {top_n} Feature Importances - {model_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.close()
        
        return importance_df
    except Exception as e:
        logger.error(f"Failed to plot feature importance for {model_name}: {e}")
        raise

def plot_shap_summary(model, X_test, feature_names, model_name, max_display=20, save_path=None):
    """Plot SHAP summary for the model"""
    try:
        # Create a figure
        plt.figure(figsize=(12, 10))
        
        # For tree-based models
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)) or \
           isinstance(model, xgb.XGBClassifier) or \
           isinstance(model, lgb.LGBMClassifier):
            explainer = shap.TreeExplainer(model)
            
            # Limit to a sample for computational efficiency
            sample_size = min(500, X_test.shape[0])
            X_sample = X_test[:sample_size]
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different formats returned by different models
            if isinstance(shap_values, list):
                # For models that return a list of shap values for each class
                class_to_show = 1  # Show SHAP values for the positive class (readmission)
                shap_vals = shap_values[class_to_show] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
            
            # Use feature names for the plot
            feature_names_truncated = [name[:30] + '...' if len(name) > 30 else name for name in feature_names]
            
            if len(feature_names_truncated) > X_sample.shape[1]:
                # Truncate feature names if needed
                feature_names_truncated = feature_names_truncated[:X_sample.shape[1]]
            elif len(feature_names_truncated) < X_sample.shape[1]:
                # Add generic names if needed
                feature_names_truncated += [f"Feature_{i}" for i in range(len(feature_names_truncated), X_sample.shape[1])]
            
            # Create SHAP summary plot
            shap.summary_plot(
                shap_vals, 
                X_sample, 
                feature_names=feature_names_truncated,
                max_display=max_display,
                show=False
            )
            
        # For linear models
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X_test)
            shap_values = explainer.shap_values(X_test)
            
            # Use feature names for the plot
            feature_names_truncated = [name[:30] + '...' if len(name) > 30 else name for name in feature_names]
            
            # Create SHAP summary plot
            shap.summary_plot(
                shap_values, 
                X_test, 
                feature_names=feature_names_truncated, 
                max_display=max_display,
                show=False
            )
        
        # For pipeline models, extract the classifier
        elif hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
            classifier = model.named_steps['classifier']
            
            if isinstance(classifier, (RandomForestClassifier, GradientBoostingClassifier)) or \
               isinstance(classifier, xgb.XGBClassifier) or \
               isinstance(classifier, lgb.LGBMClassifier):
                explainer = shap.TreeExplainer(classifier)
            elif isinstance(classifier, LogisticRegression):
                explainer = shap.LinearExplainer(classifier, X_test)
            else:
                logger.warning(f"Unsupported classifier type for SHAP: {type(classifier)}")
                return
            
            # Limit to a sample for computational efficiency
            sample_size = min(500, X_test.shape[0])
            X_sample = X_test[:sample_size]
            
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different formats returned by different models
            if isinstance(shap_values, list):
                # For models that return a list of shap values for each class
                class_to_show = 1  # Show SHAP values for the positive class (readmission)
                shap_vals = shap_values[class_to_show] if len(shap_values) > 1 else shap_values[0]
            else:
                shap_vals = shap_values
            
            # Use feature names for the plot
            feature_names_truncated = [name[:30] + '...' if len(name) > 30 else name for name in feature_names]
            
            # Create SHAP summary plot
            shap.summary_plot(
                shap_vals, 
                X_sample, 
                feature_names=feature_names_truncated, 
                max_display=max_display,
                show=False
            )
        
        else:
            logger.warning(f"Unsupported model type for SHAP: {type(model)}")
            return
        
        plt.title(f"SHAP Summary - {model_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved to {save_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot SHAP summary for {model_name}: {e}")
        raise

def create_model_comparison_table(results_dict, save_path=None):
    """Create a table comparing model performance metrics"""
    try:
        # Extract metrics from results
        model_names = []
        metrics = {
            'Accuracy': [],
            'Precision': [],
            'Recall': [],
            'F1 Score': [],
            'ROC AUC': [],
            'Avg Precision': []
        }
        
        for model_name, results in results_dict.items():
            model_names.append(model_name)
            metrics['Accuracy'].append(results['accuracy'])
            metrics['Precision'].append(results['precision'])
            metrics['Recall'].append(results['recall'])
            metrics['F1 Score'].append(results['f1_score'])
            metrics['ROC AUC'].append(results['roc_auc'])
            metrics['Avg Precision'].append(results['avg_precision'])
        
        # Create DataFrame
        comparison_df = pd.DataFrame(metrics, index=model_names)
        
        # Format as percentages
        comparison_df = comparison_df.apply(lambda x: x.map('{:.2%}'.format))
        
        # Save to CSV
        if save_path:
            comparison_df.to_csv(save_path)
            logger.info(f"Model comparison table saved to {save_path}")
        
        return comparison_df
    except Exception as e:
        logger.error(f"Failed to create model comparison table: {e}")
        raise

def save_model(model, model_name):
    """Save the trained model to disk"""
    try:
        # Create model file path
        model_path = MODELS_DIR / f"{model_name}.pkl"
        
        # Save the model
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    except Exception as e:
        logger.error(f"Failed to save model {model_name}: {e}")
        raise

def main():
    """Main function to orchestrate model training and evaluation"""
    try:
        # Create directories
        create_directories()
        
        # Load processed data
        processed_data = load_processed_data()
        X_train = processed_data['X_train']
        X_test = processed_data['X_test']
        y_train = processed_data['y_train']
        y_test = processed_data['y_test']
        feature_names = processed_data['feature_names']
        
        logger.info(f"Loaded data with {X_train.shape[0]} training samples and {X_test.shape[0]} test samples")
        logger.info(f"Class distribution in training set: {np.bincount(y_train)}")
        logger.info(f"Class distribution in test set: {np.bincount(y_test)}")
        
        # Train models
        models = {
            'Logistic Regression': train_logistic_regression(X_train, y_train),
            'Random Forest': train_random_forest(X_train, y_train),
            'Gradient Boosting': train_gradient_boosting(X_train, y_train),
            'XGBoost': train_xgboost(X_train, y_train),
            'LightGBM': train_lightgbm(X_train, y_train)
        }
        
        # Also train a model with SMOTE to handle class imbalance
        models['XGBoost with SMOTE'] = train_with_smote(
            xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_child_weight=2,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            X_train, y_train
        )
        
        # Evaluate models
        results_dict = {}
        for model_name, model in models.items():
            results = evaluate_model(model, X_test, y_test, model_name)
            results_dict[model_name] = results
            
            # Save the model
            save_model(model, model_name.replace(' ', '_').lower())
        
        # Create comparison plots and tables
        plot_roc_curve(results_dict, save_path=FIGURES_DIR / "roc_curves.png")
        plot_precision_recall_curve(results_dict, save_path=FIGURES_DIR / "pr_curves.png")
        plot_confusion_matrices(results_dict, save_path=FIGURES_DIR / "confusion_matrices.png")
        
        # Create comparison table
        comparison_df = create_model_comparison_table(
            results_dict, 
            save_path=RESULTS_DIR / "model_comparison.csv"
        )
        
        # Plot feature importance for the best model
        # Find the best model based on F1 score
        best_model_name = max(results_dict.items(), key=lambda x: x[1]['f1_score'])[0]
        best_model = models[best_model_name]
        
        logger.info(f"Best model based on F1 score: {best_model_name}")
        
        # Check if the best model is a pipeline with SMOTE
        if hasattr(best_model, 'named_steps') and 'classifier' in best_model.named_steps:
            plot_feature_importance(
                best_model.named_steps['classifier'], 
                feature_names, 
                best_model_name,
                save_path=FIGURES_DIR / "feature_importance.png"
            )
            
            # Plot SHAP values for the best model
            plot_shap_summary(
                best_model.named_steps['classifier'], 
                X_test, 
                feature_names, 
                best_model_name,
                save_path=FIGURES_DIR / "shap_summary.png"
            )
        else:
            plot_feature_importance(
                best_model, 
                feature_names, 
                best_model_name,
                save_path=FIGURES_DIR / "feature_importance.png"
            )
            
            # Plot SHAP values for the best model
            plot_shap_summary(
                best_model, 
                X_test, 
                feature_names, 
                best_model_name,
                save_path=FIGURES_DIR / "shap_summary.png"
            )
        
        logger.info("Model training and evaluation complete")
        
        return models, results_dict, comparison_df
    
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise

if __name__ == "__main__":
    main()
