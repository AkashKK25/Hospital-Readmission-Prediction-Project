"""
Business Impact Analysis for Hospital Readmission Prediction
Analyzes the potential business value of the readmission prediction model
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from sklearn.metrics import confusion_matrix

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

def load_model_and_data():
    """Load the best model and test data"""
    try:
        # Load processed data
        processed_data = joblib.load(PROCESSED_DATA_DIR / "model_ready_data.pkl")
        X_test = processed_data['X_test']
        y_test = processed_data['y_test']
        
        # Load the original processed dataframe for additional analysis
        df = pd.read_csv(PROCESSED_DATA_DIR / "diabetes_processed.csv")
        
        # Determine the best model (use XGBoost with SMOTE as a default)
        model_path = MODELS_DIR / "xgboost_with_smote.pkl"
        
        if not model_path.exists():
            # Fall back to regular XGBoost if SMOTE version doesn't exist
            model_path = MODELS_DIR / "xgboost.pkl"
            
        if not model_path.exists():
            # Fall back to Random Forest if XGBoost doesn't exist
            model_path = MODELS_DIR / "random_forest.pkl"
        
        model = joblib.load(model_path)
        logger.info(f"Loaded model from {model_path}")
        
        return model, X_test, y_test, df
    except Exception as e:
        logger.error(f"Failed to load model and data: {e}")
        raise

def calculate_readmission_costs(model, X_test, y_test, base_readmission_cost=15000):
    """
    Calculate costs related to readmissions and potential savings
    
    Parameters:
    - model: Trained prediction model
    - X_test: Test feature data
    - y_test: True readmission labels
    - base_readmission_cost: Average cost of a readmission (USD)
    
    Returns:
    - Dictionary of cost metrics
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        y_prob = None
        
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'predict_proba'):
            y_prob = model.named_steps['classifier'].predict_proba(X_test)[:, 1]
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate basic metrics
        total_patients = len(y_test)
        actual_readmissions = np.sum(y_test == 1)
        predicted_readmissions = np.sum(y_pred == 1)
        true_positives = tp  # Correctly identified readmissions
        false_positives = fp  # Incorrectly flagged as readmissions
        false_negatives = fn  # Missed readmissions
        
        # Cost calculations
        total_readmission_cost = actual_readmissions * base_readmission_cost
        
        # Assuming intervention costs and effectiveness
        intervention_cost_per_patient = 500  # Cost of intervention program per patient
        intervention_effectiveness = 0.40  # Reduction in readmission probability
        
        # Calculate costs and savings under different scenarios
        
        # Scenario 1: No model, no interventions
        scenario_1_cost = total_readmission_cost
        
        # Scenario 2: No model, intervene on all patients
        scenario_2_intervention_cost = total_patients * intervention_cost_per_patient
        scenario_2_readmission_cost = actual_readmissions * (1 - intervention_effectiveness) * base_readmission_cost
        scenario_2_total_cost = scenario_2_intervention_cost + scenario_2_readmission_cost
        scenario_2_savings = scenario_1_cost - scenario_2_total_cost
        
        # Scenario 3: Use model, intervene on predicted positives
        scenario_3_intervention_cost = predicted_readmissions * intervention_cost_per_patient
        
        # Readmissions after intervention (adjusted for true positives only)
        prevented_readmissions = true_positives * intervention_effectiveness
        remaining_readmissions = actual_readmissions - prevented_readmissions
        scenario_3_readmission_cost = remaining_readmissions * base_readmission_cost
        scenario_3_total_cost = scenario_3_intervention_cost + scenario_3_readmission_cost
        scenario_3_savings = scenario_1_cost - scenario_3_total_cost
        
        # Scenario 4: Perfect prediction, intervene on actual positives
        scenario_4_intervention_cost = actual_readmissions * intervention_cost_per_patient
        scenario_4_readmission_cost = actual_readmissions * (1 - intervention_effectiveness) * base_readmission_cost
        scenario_4_total_cost = scenario_4_intervention_cost + scenario_4_readmission_cost
        scenario_4_savings = scenario_1_cost - scenario_4_total_cost
        
        # Calculate ROI for model-based intervention (Scenario 3)
        roi = (scenario_3_savings / scenario_3_intervention_cost) * 100 if scenario_3_intervention_cost > 0 else 0
        
        # Calculate cost metrics
        cost_per_prevented_readmission = scenario_3_intervention_cost / prevented_readmissions if prevented_readmissions > 0 else float('inf')
        net_savings_per_patient = scenario_3_savings / total_patients
        
        # Store results
        results = {
            'total_patients': total_patients,
            'actual_readmissions': actual_readmissions,
            'readmission_rate': actual_readmissions / total_patients,
            'predicted_readmissions': predicted_readmissions,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'base_readmission_cost': base_readmission_cost,
            'intervention_cost': intervention_cost_per_patient,
            'intervention_effectiveness': intervention_effectiveness,
            'prevented_readmissions': prevented_readmissions,
            'scenario_1_cost': scenario_1_cost,
            'scenario_2_total_cost': scenario_2_total_cost,
            'scenario_2_savings': scenario_2_savings,
            'scenario_3_total_cost': scenario_3_total_cost,
            'scenario_3_savings': scenario_3_savings,
            'scenario_4_total_cost': scenario_4_total_cost,
            'scenario_4_savings': scenario_4_savings,
            'roi': roi,
            'cost_per_prevented_readmission': cost_per_prevented_readmission,
            'net_savings_per_patient': net_savings_per_patient
        }
        
        # Log key metrics
        logger.info(f"Business impact analysis results:")
        logger.info(f"  Actual readmission rate: {results['readmission_rate']:.2%}")
        logger.info(f"  Predicted readmissions: {predicted_readmissions} out of {total_patients} patients")
        logger.info(f"  Potential prevented readmissions: {prevented_readmissions:.1f}")
        logger.info(f"  Potential savings with model-based intervention: ${scenario_3_savings:,.2f}")
        logger.info(f"  ROI for model-based intervention: {roi:.1f}%")
        
        return results
    except Exception as e:
        logger.error(f"Failed to calculate readmission costs: {e}")
        raise

def analyze_high_risk_profiles(df, model, X_test, y_test, threshold=0.7):
    """
    Analyze characteristics of high-risk patients
    
    Parameters:
    - df: Original processed dataframe
    - model: Trained prediction model
    - X_test: Test feature data
    - y_test: True readmission labels
    - threshold: Probability threshold for high risk designation
    
    Returns:
    - DataFrame with high risk profiles
    """
    try:
        # Get probability predictions if available
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps.get('classifier', None), 'predict_proba'):
            y_prob = model.named_steps['classifier'].predict_proba(X_test)[:, 1]
        else:
            logger.warning("Model does not support probability predictions, using binary predictions instead")
            y_prob = model.predict(X_test)
        
        # Create a DataFrame with risk scores
        risk_df = pd.DataFrame({
            'actual_readmission': y_test,
            'risk_score': y_prob
        })
        
        # Identify high-risk patients
        high_risk = risk_df[risk_df['risk_score'] >= threshold]
        high_risk_count = len(high_risk)
        high_risk_actual_readmits = high_risk['actual_readmission'].sum()
        
        logger.info(f"Identified {high_risk_count} high-risk patients (score >= {threshold})")
        logger.info(f"Of these, {high_risk_actual_readmits} were actually readmitted")
        logger.info(f"Precision in high-risk group: {high_risk_actual_readmits / high_risk_count if high_risk_count > 0 else 0:.2%}")
        
        # Analyze risk thresholds
        risk_thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_metrics = []
        
        for thresh in risk_thresholds:
            patients_above_threshold = np.sum(y_prob >= thresh)
            true_positives_above_threshold = np.sum((y_prob >= thresh) & (y_test == 1))
            precision_at_threshold = true_positives_above_threshold / patients_above_threshold if patients_above_threshold > 0 else 0
            recall_at_threshold = true_positives_above_threshold / np.sum(y_test == 1) if np.sum(y_test == 1) > 0 else 0
            
            threshold_metrics.append({
                'threshold': thresh,
                'patients_flagged': patients_above_threshold,
                'true_readmissions': true_positives_above_threshold,
                'precision': precision_at_threshold,
                'recall': recall_at_threshold
            })
        
        threshold_df = pd.DataFrame(threshold_metrics)
        
        # If the original dataframe has the right index alignment, we can analyze high risk factors
        if len(df) >= len(X_test) and 'readmitted_30d' in df.columns:
            # Try to align indices if possible
            try:
                # Extract the indices of high-risk patients in the test set
                test_indices = np.where(y_prob >= threshold)[0]
                
                # Check if we have enough information to map these to the original dataframe
                # This is a simplification and might not work for all cases
                high_risk_profiles = df.iloc[-len(X_test):].iloc[test_indices]
                
                # Analyze characteristics of high-risk patients
                hr_analysis = {
                    'age_distribution': high_risk_profiles['age'].value_counts().to_dict() if 'age' in high_risk_profiles.columns else None,
                    'gender_distribution': high_risk_profiles['gender'].value_counts().to_dict() if 'gender' in high_risk_profiles.columns else None,
                    'avg_time_in_hospital': high_risk_profiles['time_in_hospital'].mean() if 'time_in_hospital' in high_risk_profiles.columns else None,
                    'avg_num_medications': high_risk_profiles['num_medications'].mean() if 'num_medications' in high_risk_profiles.columns else None,
                    'avg_num_procedures': high_risk_profiles['num_procedures'].mean() if 'num_procedures' in high_risk_profiles.columns else None,
                    'primary_diagnosis': high_risk_profiles['diag_1_category'].value_counts().to_dict() if 'diag_1_category' in high_risk_profiles.columns else None
                }
                
                logger.info("High-risk patient profile analysis completed")
                
                return {
                    'high_risk_patients': high_risk,
                    'threshold_analysis': threshold_df,
                    'risk_profile_analysis': hr_analysis,
                    'high_risk_profiles': high_risk_profiles
                }
            except Exception as e:
                logger.warning(f"Could not link high-risk indices to original dataframe: {e}")
                return {
                    'high_risk_patients': high_risk,
                    'threshold_analysis': threshold_df
                }
        else:
            logger.warning("Original dataframe does not align with test set, cannot analyze high-risk profiles")
            return {
                'high_risk_patients': high_risk,
                'threshold_analysis': threshold_df
            }
    except Exception as e:
        logger.error(f"Failed to analyze high-risk profiles: {e}")
        raise

def calculate_intervention_optimization(cost_results, risk_analysis):
    """
    Optimize intervention strategies based on cost-benefit analysis
    
    Parameters:
    - cost_results: Results from calculate_readmission_costs
    - risk_analysis: Results from analyze_high_risk_profiles
    
    Returns:
    - Dictionary with optimization results
    """
    try:
        # Get threshold analysis
        threshold_df = risk_analysis['threshold_analysis']
        
        # Calculate the expected net benefit at each threshold
        intervention_cost = cost_results['intervention_cost']
        readmission_cost = cost_results['base_readmission_cost']
        intervention_effectiveness = cost_results['intervention_effectiveness']
        
        # Calculate the net benefit for each threshold
        threshold_df['intervention_cost_total'] = threshold_df['patients_flagged'] * intervention_cost
        threshold_df['prevented_readmissions'] = threshold_df['true_readmissions'] * intervention_effectiveness
        threshold_df['savings_from_prevention'] = threshold_df['prevented_readmissions'] * readmission_cost
        threshold_df['net_benefit'] = threshold_df['savings_from_prevention'] - threshold_df['intervention_cost_total']
        threshold_df['roi'] = (threshold_df['net_benefit'] / threshold_df['intervention_cost_total']) * 100
        
        # Find the optimal threshold (maximum net benefit)
        optimal_threshold_row = threshold_df.loc[threshold_df['net_benefit'].idxmax()]
        optimal_threshold = optimal_threshold_row['threshold']
        optimal_net_benefit = optimal_threshold_row['net_benefit']
        optimal_roi = optimal_threshold_row['roi']
        
        logger.info(f"Optimal risk threshold: {optimal_threshold:.2f}")
        logger.info(f"At this threshold, {optimal_threshold_row['patients_flagged']} patients would receive intervention")
        logger.info(f"Expected net benefit: ${optimal_net_benefit:,.2f}")
        logger.info(f"Expected ROI: {optimal_roi:.1f}%")
        
        # Calculate benefits per hospital size
        hospital_sizes = {
            'small': 100,
            'medium': 500,
            'large': 2000,
            'system': 10000
        }
        
        hospital_benefits = {}
        
        for size_name, patient_count in hospital_sizes.items():
            # Scale optimal results to hospital size
            scale_factor = patient_count / cost_results['total_patients']
            scaled_benefit = optimal_net_benefit * scale_factor
            scaled_patients_flagged = int(optimal_threshold_row['patients_flagged'] * scale_factor)
            scaled_prevented = optimal_threshold_row['prevented_readmissions'] * scale_factor
            
            hospital_benefits[size_name] = {
                'annual_patients': patient_count,
                'intervention_patients': scaled_patients_flagged,
                'prevented_readmissions': scaled_prevented,
                'net_annual_benefit': scaled_benefit
            }
            
            logger.info(f"For a {size_name} hospital with {patient_count} annual patients:")
            logger.info(f"  Intervene on {scaled_patients_flagged} high-risk patients")
            logger.info(f"  Prevent ~{scaled_prevented:.1f} readmissions")
            logger.info(f"  Net annual benefit: ${scaled_benefit:,.2f}")
        
        # Return optimization results
        return {
            'threshold_optimization': threshold_df,
            'optimal_threshold': optimal_threshold,
            'optimal_net_benefit': optimal_net_benefit,
            'optimal_roi': optimal_roi,
            'hospital_size_benefits': hospital_benefits
        }
    except Exception as e:
        logger.error(f"Failed to calculate intervention optimization: {e}")
        raise

def plot_cost_benefit_analysis(optimization_results, save_path=None):
    """
    Plot cost-benefit analysis visualizations
    
    Parameters:
    - optimization_results: Results from calculate_intervention_optimization
    - save_path: Path to save the plots
    """
    try:
        threshold_df = optimization_results['threshold_optimization']
        optimal_threshold = optimization_results['optimal_threshold']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Cost-Benefit by Threshold
        ax1.plot(threshold_df['threshold'], threshold_df['intervention_cost_total'], 
                 label='Intervention Cost', color='red', marker='o')
        ax1.plot(threshold_df['threshold'], threshold_df['savings_from_prevention'], 
                 label='Expected Savings', color='green', marker='s')
        ax1.plot(threshold_df['threshold'], threshold_df['net_benefit'], 
                 label='Net Benefit', color='blue', marker='^', linewidth=2)
        
        # Mark the optimal threshold
        optimal_row = threshold_df[threshold_df['threshold'] == optimal_threshold].iloc[0]
        ax1.axvline(x=optimal_threshold, color='k', linestyle='--', alpha=0.7)
        ax1.text(optimal_threshold + 0.02, optimal_row['net_benefit'] * 0.8, 
                f'Optimal threshold: {optimal_threshold:.2f}', fontsize=9)
        
        ax1.set_xlabel('Risk Score Threshold')
        ax1.set_ylabel('Amount ($)')
        ax1.set_title('Cost-Benefit Analysis by Risk Threshold')
        ax1.grid(alpha=0.3)
        ax1.legend()
        
        # Plot 2: ROI by Threshold
        ax2.plot(threshold_df['threshold'], threshold_df['roi'], 
                 color='purple', marker='d', linewidth=2)
        
        # Mark the optimal threshold
        ax2.axvline(x=optimal_threshold, color='k', linestyle='--', alpha=0.7)
        
        ax2.set_xlabel('Risk Score Threshold')
        ax2.set_ylabel('Return on Investment (%)')
        ax2.set_title('ROI by Risk Threshold')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Cost-benefit analysis plot saved to {save_path}")
        
        plt.close()
        
        # Create hospital size comparison plot
        hospital_benefits = optimization_results['hospital_size_benefits']
        
        # Extract data for plotting
        sizes = list(hospital_benefits.keys())
        net_benefits = [hospital_benefits[size]['net_annual_benefit'] for size in sizes]
        prevented = [hospital_benefits[size]['prevented_readmissions'] for size in sizes]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Net Benefits by Hospital Size
        ax1.bar(sizes, net_benefits, color='green')
        
        # Add value labels
        for i, v in enumerate(net_benefits):
            ax1.text(i, v + v*0.02, f'${v:,.0f}', ha='center', fontsize=9)
        
        ax1.set_xlabel('Hospital Size')
        ax1.set_ylabel('Net Annual Benefit ($)')
        ax1.set_title('Expected Net Benefit by Hospital Size')
        
        # Plot 2: Prevented Readmissions by Hospital Size
        ax2.bar(sizes, prevented, color='blue')
        
        # Add value labels
        for i, v in enumerate(prevented):
            ax2.text(i, v + v*0.02, f'{v:.1f}', ha='center', fontsize=9)
        
        ax2.set_xlabel('Hospital Size')
        ax2.set_ylabel('Prevented Readmissions')
        ax2.set_title('Expected Prevented Readmissions by Hospital Size')
        
        plt.tight_layout()
        
        if save_path:
            hospital_size_path = save_path.parent / (save_path.stem + "_hospital_size.png")
            plt.savefig(hospital_size_path, dpi=300, bbox_inches='tight')
            logger.info(f"Hospital size comparison plot saved to {hospital_size_path}")
        
        plt.close()
    except Exception as e:
        logger.error(f"Failed to plot cost-benefit analysis: {e}")
        raise

def create_executive_summary(cost_results, risk_analysis, optimization_results):
    """
    Create an executive summary of the business impact analysis
    
    Parameters:
    - cost_results: Results from calculate_readmission_costs
    - risk_analysis: Results from analyze_high_risk_profiles
    - optimization_results: Results from calculate_intervention_optimization
    
    Returns:
    - String containing the executive summary
    """
    try:
        # Format numbers for summary
        readmission_rate = cost_results['readmission_rate'] * 100
        total_patients = cost_results['total_patients']
        actual_readmissions = cost_results['actual_readmissions']
        prevented_readmissions = cost_results['prevented_readmissions']
        model_savings = cost_results['scenario_3_savings']
        roi = cost_results['roi']
        
        optimal_threshold = optimization_results['optimal_threshold']
        optimal_net_benefit = optimization_results['optimal_net_benefit']
        optimal_roi = optimization_results['optimal_roi']
        
        # Create summary text
        summary = f"""
# Executive Summary: Hospital Readmission Prediction Business Impact

## Problem Overview
- **Current Readmission Rate:** {readmission_rate:.1f}% ({actual_readmissions} out of {total_patients} patients)
- **Average Cost Per Readmission:** ${cost_results['base_readmission_cost']:,}
- **Total Annual Readmission Cost:** ${cost_results['scenario_1_cost']:,.2f}

## Model-Based Intervention Strategy
The predictive model identifies patients at high risk of readmission, allowing targeted interventions that are more cost-effective than applying interventions universally.

- **Intervention Cost Per Patient:** ${cost_results['intervention_cost']:,}
- **Estimated Intervention Effectiveness:** {cost_results['intervention_effectiveness']*100:.0f}% reduction in readmission risk

## Key Findings

### 1. Financial Impact
- **Potential Annual Savings:** ${model_savings:,.2f} using the model-based approach
- **Return on Investment (ROI):** {roi:.1f}%
- **Net Savings Per Patient:** ${cost_results['net_savings_per_patient']:,.2f}

### 2. Optimized Risk Threshold
- **Optimal Risk Score Threshold:** {optimal_threshold:.2f}
- **Expected Net Benefit at Optimal Threshold:** ${optimal_net_benefit:,.2f}
- **Expected ROI at Optimal Threshold:** {optimal_roi:.1f}%

### 3. Clinical Impact
- **Potential Readmissions Prevented:** {prevented_readmissions:.1f} patients annually
- **Cost Per Prevented Readmission:** ${cost_results['cost_per_prevented_readmission']:,.2f}

## Projected Impact by Hospital Size
| Hospital Size | Annual Patients | Intervention Patients | Prevented Readmissions | Net Annual Benefit |
|---------------|----------------|---------------------|------------------------|-------------------|
"""
        # Add rows for each hospital size
        for size, data in optimization_results['hospital_size_benefits'].items():
            summary += f"| {size.capitalize()} | {data['annual_patients']:,} | {data['intervention_patients']:,} | {data['prevented_readmissions']:.1f} | ${data['net_annual_benefit']:,.2f} |\n"
        
        summary += """
## Recommendations
1. **Implement Risk Prediction System:** Deploy the machine learning model to identify high-risk patients at discharge
2. **Establish Targeted Intervention Program:** Focus resources on patients above the optimal risk threshold
3. **Track and Measure Outcomes:** Monitor actual readmission rates and adjust the model and thresholds as needed
4. **Expand Data Collection:** Incorporate additional variables like social determinants of health to improve prediction accuracy

The analysis demonstrates that the predictive model approach is significantly more cost-effective than either doing nothing or applying interventions universally.
"""
        
        # Save summary to file
        summary_path = RESULTS_DIR / "executive_summary.md"
        
        with open(summary_path, 'w') as f:
            f.write(summary)
        
        logger.info(f"Executive summary saved to {summary_path}")
        
        return summary
    except Exception as e:
        logger.error(f"Failed to create executive summary: {e}")
        raise

def main():
    """Main function to orchestrate business impact analysis"""
    try:
        # Create necessary directories
        for dir_path in [RESULTS_DIR, FIGURES_DIR]:
            if not dir_path.exists():
                dir_path.mkdir(parents=True)
                logger.info(f"Created directory: {dir_path}")
        
        # Load model and data
        model, X_test, y_test, df = load_model_and_data()
        
        # Calculate readmission costs
        cost_results = calculate_readmission_costs(model, X_test, y_test)
        
        # Analyze high-risk patient profiles
        risk_analysis = analyze_high_risk_profiles(df, model, X_test, y_test)
        
        # Calculate intervention optimization
        optimization_results = calculate_intervention_optimization(cost_results, risk_analysis)
        
        # Plot cost-benefit analysis
        plot_cost_benefit_analysis(optimization_results, save_path=FIGURES_DIR / "cost_benefit_analysis.png")
        
        # Create executive summary
        summary = create_executive_summary(cost_results, risk_analysis, optimization_results)
        
        logger.info("Business impact analysis complete")
        
        return {
            'cost_results': cost_results,
            'risk_analysis': risk_analysis,
            'optimization_results': optimization_results,
            'summary': summary
        }
    
    except Exception as e:
        logger.error(f"Business impact analysis failed: {e}")
        raise

if __name__ == "__main__":
    main()
