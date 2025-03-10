from setuptools import find_packages, setup

setup(
    name='hospital_readmission',
    packages=find_packages(),
    version='0.1.0',
    description='Predictive analysis of hospital readmission rates',
    author='Data Analyst',
    license='MIT',
    install_requires=[
        'numpy>=1.21.6',
        'pandas>=1.3.5',
        'scikit-learn>=1.0.2',
        'matplotlib>=3.5.2',
        'seaborn>=0.11.2',
        'plotly>=5.9.0',
        'dash>=2.6.0',
        'dash-bootstrap-components>=1.2.0',
        'shap>=0.41.0',
        'joblib>=1.1.0',
        'notebook>=6.4.12',
        'statsmodels>=0.13.2',
        'imbalanced-learn>=0.9.1',
        'xgboost>=1.6.1',
        'lightgbm>=3.3.2',
        'openpyxl>=3.0.10',
    ],
    python_requires='>=3.8',
)
