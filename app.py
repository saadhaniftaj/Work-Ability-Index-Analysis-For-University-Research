#!/usr/bin/env python3
"""
Flask Web Application for WAI Analysis
Deployable on AWS with file upload and downloadable text report generation
"""

import os
import sys
import traceback
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, send_file, jsonify, flash, redirect, url_for, session
from werkzeug.utils import secure_filename
from datetime import datetime
import tempfile
import zipfile
from io import BytesIO
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, pearsonr, f_oneway
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import warnings
import json
warnings.filterwarnings('ignore')

import logging
from logging.handlers import RotatingFileHandler

# Configure logging
if not os.path.exists('logs'):
    os.mkdir('logs')
file_handler = RotatingFileHandler('logs/wai_analysis.log', maxBytes=10240, backupCount=10)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
file_handler.setLevel(logging.INFO)

app = Flask(__name__)
app.secret_key = 'wai_analysis_secret_key_2025'
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Add console handler for AWS logs
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
))
app.logger.addHandler(console_handler)

app.logger.info('WAI Analysis startup')

# Configuration
UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['REPORTS_FOLDER'] = REPORTS_FOLDER

# Create upload and reports folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

app.logger.info(f'Upload folder: {UPLOAD_FOLDER}')
app.logger.info(f'Reports folder: {app.config["REPORTS_FOLDER"]}')

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class AccurateWAIAnalyzer:
    """Accurate WAI Analysis class with proper Excel handling"""
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.results = {}
        
    def load_and_clean_data(self):
        """Load and clean the Excel data with proper header handling"""
        try:
            # Read Excel file with header in row 3 (index 2)
            self.data = pd.read_excel(self.file_path, header=2)
            app.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            app.logger.info(f"Columns: {list(self.data.columns)}")
            
            # Basic cleaning
            self.data = self.data.dropna(how='all')
            
            # Map the actual column names to standard names
            column_mapping = {
                'WAI score': 'work_ability_index',
                'GAD-7': 'gad_7',
                'Social Support': 'social_support_scale',
                'BRS': 'brs_scale',
                'Grip R': 'grip_strength_right',
                'Grip L': 'grip_strength_left',
                'Pain Level 1-10': 'pain',
                'BP Final': 'blood_pressure',
                'Weight': 'weight',
                'Sex': 'gender',
                'DOB': 'date_of_birth',
                'Date of Injury': 'date_of_injury',
                'BODY PART': 'type_of_injury',
                'RTW in FCE': 'return_to_work',
                'Current Work Status': 'current_work_status',
                'Height': 'height',
                'BMI': 'bmi',
                'Formula': 'formula',
                'WC or Disability': 'compensation_type',
                'Language': 'language',
                'Using Xcelable App': 'app_usage',
                'Therapist': 'therapist',
                'PT EMAIL': 'pt_email',
                'Report Upload': 'report_upload',
                'Email Address': 'email_address',
                'DEMOS': 'demos'
            }
            
            # Rename columns
            for old_name, new_name in column_mapping.items():
                if old_name in self.data.columns:
                    self.data[new_name] = self.data[old_name]
            
            # Clean numeric columns
            numeric_columns = ['work_ability_index', 'gad_7', 'social_support_scale', 'brs_scale', 
                             'grip_strength_right', 'grip_strength_left', 'pain', 'weight', 'height', 'bmi']
            
            for col in numeric_columns:
                if col in self.data.columns:
                    self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                    
            # Clean blood pressure data
            if 'blood_pressure' in self.data.columns:
                self.data['systolic'] = np.nan
                self.data['diastolic'] = np.nan
                
                for idx, bp_value in self.data['blood_pressure'].items():
                    if pd.notna(bp_value):
                        bp_str = str(bp_value).strip()
                        # Handle various BP formats
                        if '/' in bp_str:
                            parts = bp_str.split('/')
                            if len(parts) == 2:
                                try:
                                    self.data.at[idx, 'systolic'] = float(parts[0].strip())
                                    self.data.at[idx, 'diastolic'] = float(parts[1].strip())
                                except:
                                    pass
            
            # Clean gender data
            if 'gender' in self.data.columns:
                self.data['gender'] = self.data['gender'].astype(str).str.lower().str.strip()
                self.data['gender'] = self.data['gender'].replace({
                    'm': 'male', 'male': 'male',
                    'f': 'female', 'female': 'female',
                    '1': 'male', '2': 'female'
                })
            
            # Clean return to work data
            if 'return_to_work' in self.data.columns:
                self.data['return_to_work'] = self.data['return_to_work'].astype(str).str.lower().str.strip()
                self.data['rtw_binary'] = self.data['return_to_work'].str.contains('yes|ft|full|part', case=False, na=False)
            
            # Calculate age from DOB
            if 'date_of_birth' in self.data.columns:
                try:
                    self.data['age'] = (pd.Timestamp.now() - pd.to_datetime(self.data['date_of_birth'])).dt.total_seconds() / (365.25 * 24 * 60 * 60)
                except:
                    self.data['age'] = None
            
            # Calculate days since injury
            if 'date_of_injury' in self.data.columns:
                try:
                    self.data['days_since_injury'] = (pd.Timestamp.now() - pd.to_datetime(self.data['date_of_injury'])).dt.total_seconds() / (24 * 60 * 60)
                except:
                    self.data['days_since_injury'] = None
            
            app.logger.info("Data cleaning completed.")
            return True
            
        except Exception as e:
            app.logger.error(f"Error loading data: {e}")
            return False
    
    def requirement_1_multiple_regression(self):
        """Requirement 1: Multiple regression analysis"""
        app.logger.info("=== REQUIREMENT 1: Multiple Regression Analysis ===")
        
        # Prepare data for multiple regression
        regression_data = self.data[['work_ability_index', 'gad_7', 'social_support_scale', 'brs_scale']].dropna()
        
        if len(regression_data) < 10:
            self.results['requirement_1'] = "Insufficient data for multiple regression analysis"
            app.logger.warning(f"Insufficient data: only {len(regression_data)} complete cases")
            return
        
        X = regression_data[['gad_7', 'social_support_scale', 'brs_scale']]
        y = regression_data['work_ability_index']
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Fit multiple regression
        model = LinearRegression()
        model.fit(X_scaled, y)
        
        # Calculate statistics
        y_pred = model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        
        # Individual correlations
        correlations = {}
        for col in ['gad_7', 'social_support_scale', 'brs_scale']:
            corr, p_val = pearsonr(regression_data['work_ability_index'], regression_data[col])
            correlations[col] = {'correlation': corr, 'p_value': p_val}
        
        self.results['requirement_1'] = {
            'n_samples': len(regression_data),
            'r2_score': r2,
            'coefficients': dict(zip(['gad_7', 'social_support_scale', 'brs_scale'], model.coef_)),
            'intercept': model.intercept_,
            'individual_correlations': correlations
        }
        
        app.logger.info(f"Multiple Regression Results:")
        app.logger.info(f"Sample size: {len(regression_data)}")
        app.logger.info(f"R² Score: {r2:.4f}")
        app.logger.info(f"Intercept: {model.intercept_:.4f}")
        app.logger.info(f"Coefficients: GAD-7={model.coef_[0]:.4f}, Social Support={model.coef_[1]:.4f}, BRS={model.coef_[2]:.4f}")
    
    def requirement_2_t_tests(self):
        """Requirement 2: T-tests with each instrument"""
        app.logger.info("=== REQUIREMENT 2: T-Test Analysis ===")
        
        instruments = ['gad_7', 'social_support_scale', 'brs_scale']
        t_test_results = {}
        
        for instrument in instruments:
            if instrument not in self.data.columns:
                continue
            
            # Prepare data
            test_data = self.data[['work_ability_index', instrument]].dropna()
            
            if len(test_data) < 10:
                t_test_results[instrument] = "Insufficient data"
                continue
            
            # Split into high/low groups based on median
            median_wai = test_data['work_ability_index'].median()
            high_wai = test_data[test_data['work_ability_index'] > median_wai][instrument]
            low_wai = test_data[test_data['work_ability_index'] <= median_wai][instrument]
            
            if len(high_wai) < 5 or len(low_wai) < 5:
                t_test_results[instrument] = "Insufficient data for group comparison"
                continue
            
            # Perform t-test
            t_stat, p_val = ttest_ind(high_wai, low_wai)
            
            t_test_results[instrument] = {
                't_statistic': t_stat,
                'p_value': p_val,
                'high_wai_mean': high_wai.mean(),
                'low_wai_mean': low_wai.mean(),
                'high_wai_n': len(high_wai),
                'low_wai_n': len(low_wai)
            }
        
        self.results['requirement_2'] = t_test_results
    
    def requirement_3_wai_below_27_rtw(self):
        """Requirement 3: WAI < 27 and return to work analysis"""
        app.logger.info("=== REQUIREMENT 3: WAI < 27 and Return to Work Analysis ===")
        
        # Filter for WAI < 27
        low_wai_data = self.data[self.data['work_ability_index'] < 27].copy()
        
        if len(low_wai_data) == 0:
            self.results['requirement_3'] = "No participants with WAI < 27"
            return
        
        total_participants = len(self.data.dropna(subset=['work_ability_index']))
        low_wai_count = len(low_wai_data)
        low_wai_percentage = (low_wai_count / total_participants) * 100
        
        # Return to work analysis
        rtw_data = low_wai_data.dropna(subset=['rtw_binary'])
        if len(rtw_data) == 0:
            rtw_percentage = 0
            rtw_count = 0
        else:
            rtw_count = rtw_data['rtw_binary'].sum()
            rtw_percentage = (rtw_count / len(rtw_data)) * 100
                    
        self.results['requirement_3'] = {
            'total_participants': total_participants,
            'low_wai_count': low_wai_count,
            'low_wai_percentage': low_wai_percentage,
            'rtw_count': rtw_count,
            'rtw_percentage': rtw_percentage,
            'low_wai_with_rtw_data': len(rtw_data)
        }
    
    def requirement_4_demographic_factors(self):
        """Requirement 4: Effect of demographic factors on work ability"""
        app.logger.info("=== REQUIREMENT 4: Demographic Factors Analysis ===")
        
        demographic_results = {}
        
        # Weight analysis
        if 'weight' in self.data.columns:
            weight_data = self.data[['work_ability_index', 'weight']].dropna()
            if len(weight_data) >= 10:
                corr, p_val = pearsonr(weight_data['work_ability_index'], weight_data['weight'])
                demographic_results['weight'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(weight_data)
                }
        
        # Gender analysis
        if 'gender' in self.data.columns:
            gender_data = self.data[['work_ability_index', 'gender']].dropna()
            gender_data = gender_data[gender_data['gender'].isin(['male', 'female'])]
            
            if len(gender_data) >= 10:
                male_wai = gender_data[gender_data['gender'] == 'male']['work_ability_index']
                female_wai = gender_data[gender_data['gender'] == 'female']['work_ability_index']
                
                if len(male_wai) >= 5 and len(female_wai) >= 5:
                    t_stat, p_val = ttest_ind(male_wai, female_wai)
                    demographic_results['gender'] = {
                        't_statistic': t_stat,
                        'p_value': p_val,
                        'male_mean': male_wai.mean(),
                        'female_mean': female_wai.mean(),
                        'male_n': len(male_wai),
                        'female_n': len(female_wai)
                    }
        
        # Age analysis
        if 'age' in self.data.columns:
            age_data = self.data[['work_ability_index', 'age']].dropna()
            if len(age_data) >= 10:
                corr, p_val = pearsonr(age_data['work_ability_index'], age_data['age'])
                demographic_results['age'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(age_data)
                }
        
        # Date of injury analysis
        if 'days_since_injury' in self.data.columns:
            doi_data = self.data[['work_ability_index', 'days_since_injury']].dropna()
            if len(doi_data) >= 10:
                corr, p_val = pearsonr(doi_data['work_ability_index'], doi_data['days_since_injury'])
                demographic_results['days_since_injury'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(doi_data)
                }
            
        # Type of injury analysis
        if 'type_of_injury' in self.data.columns:
            injury_data = self.data[['work_ability_index', 'type_of_injury']].dropna()
            if len(injury_data) >= 10:
                # Get unique injury types with sufficient data
                injury_counts = injury_data['type_of_injury'].value_counts()
                valid_types = injury_counts[injury_counts >= 5].index
                
                if len(valid_types) >= 2:
                    injury_groups = []
                    for injury_type in valid_types:
                        group_data = injury_data[injury_data['type_of_injury'] == injury_type]['work_ability_index']
                        injury_groups.append(group_data)
                    
                    if len(injury_groups) >= 2:
                        f_stat, p_val = f_oneway(*injury_groups)
                        demographic_results['type_of_injury'] = {
                            'f_statistic': f_stat,
                            'p_value': p_val,
                            'group_means': {valid_types[i]: group.mean() for i, group in enumerate(injury_groups)},
                            'group_counts': {valid_types[i]: len(group) for i, group in enumerate(injury_groups)}
                        }
        
        self.results['requirement_4'] = demographic_results
    
    def requirement_5_pain_bp_effects(self):
        """Requirement 5: Effect of pain and blood pressure on work ability"""
        app.logger.info("=== REQUIREMENT 5: Pain and Blood Pressure Effects ===")
        
        pain_bp_results = {}
        
        # Pain analysis
        if 'pain' in self.data.columns:
            pain_data = self.data[['work_ability_index', 'pain']].dropna()
            if len(pain_data) >= 10:
                corr, p_val = pearsonr(pain_data['work_ability_index'], pain_data['pain'])
                pain_bp_results['pain'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(pain_data)
                }
            
        # Blood pressure analysis
        bp_results = {}
        if 'systolic' in self.data.columns and 'diastolic' in self.data.columns:
            # Systolic BP
            systolic_data = self.data[['work_ability_index', 'systolic']].dropna()
            if len(systolic_data) >= 10:
                corr, p_val = pearsonr(systolic_data['work_ability_index'], systolic_data['systolic'])
                bp_results['systolic'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(systolic_data)
                }
            
            # Diastolic BP
            diastolic_data = self.data[['work_ability_index', 'diastolic']].dropna()
            if len(diastolic_data) >= 10:
                corr, p_val = pearsonr(diastolic_data['work_ability_index'], diastolic_data['diastolic'])
                bp_results['diastolic'] = {
                    'correlation': corr,
                    'p_value': p_val,
                    'n_samples': len(diastolic_data)
                }
        
        pain_bp_results['blood_pressure'] = bp_results
        self.results['requirement_5'] = pain_bp_results
    
    def requirement_6_grip_strength_correlation(self):
        """Requirement 6: Correlation of grip strength and work ability"""
        app.logger.info("=== REQUIREMENT 6: Grip Strength Correlation ===")
        
        if 'grip_strength_right' not in self.data.columns and 'grip_strength_left' not in self.data.columns:
            self.results['requirement_6'] = "Grip strength data not available"
            return
        
        grip_results = {}
        # Analyze right grip strength if available
        if 'grip_strength_right' in self.data.columns:
            grip_data_right = self.data[['work_ability_index', 'grip_strength_right']].dropna()
            if len(grip_data_right) >= 10:
                corr_r, p_val_r = pearsonr(grip_data_right['work_ability_index'], grip_data_right['grip_strength_right'])
                grip_results['right_grip'] = {
                    'correlation': corr_r,
                    'p_value': p_val_r,
                    'n_samples': len(grip_data_right),
                    'wai_mean': grip_data_right['work_ability_index'].mean(),
                    'grip_mean': grip_data_right['grip_strength_right'].mean()
                }
        
        # Analyze left grip strength if available
        if 'grip_strength_left' in self.data.columns:
            grip_data_left = self.data[['work_ability_index', 'grip_strength_left']].dropna()
            if len(grip_data_left) >= 10:
                corr_l, p_val_l = pearsonr(grip_data_left['work_ability_index'], grip_data_left['grip_strength_left'])
                grip_results['left_grip'] = {
                    'correlation': corr_l,
                    'p_value': p_val_l,
                    'n_samples': len(grip_data_left),
                    'wai_mean': grip_data_left['work_ability_index'].mean(),
                    'grip_mean': grip_data_left['grip_strength_left'].mean()
                }
        
        if not grip_results:
            self.results['requirement_6'] = "No sufficient grip strength data for analysis"
        else:
            self.results['requirement_6'] = grip_results
    
    def run_complete_analysis(self):
        """Run the complete analysis"""
        app.logger.info("Starting complete analysis...")
        if not self.load_and_clean_data():
            return False
        
        # Check if data was loaded successfully and is not empty
        if self.data is None or self.data.empty:
            app.logger.error("No data available for analysis after loading and cleaning.")
            self.results['error'] = "No data available for analysis. Please check the uploaded file and column mappings."
            return False

        self.requirement_1_multiple_regression()
        self.requirement_2_t_tests()
        self.requirement_3_wai_below_27_rtw()
        self.requirement_4_demographic_factors()
        self.requirement_5_pain_bp_effects()
        self.requirement_6_grip_strength_correlation()
        
        return True

def generate_report(results_list):
    """Generate comprehensive analysis report as a string"""
    report_content = []
    
    report_content.append("="*80)
    report_content.append("COMPREHENSIVE WORK ABILITY INDEX ANALYSIS REPORT")
    report_content.append("="*80)
    
    for i, results in enumerate(results_list, 1):
        file_name = results.get('file_name', f'File {i}')
        report_content.append(f"\n\n=== ANALYSIS FOR: {file_name} ===")
        report_content.append(f"Analysis Date: {results.get('analysis_date', 'N/A')}")
        
        if 'error' in results:
            report_content.append(f"❌ Error during analysis: {results['error']}")
            continue

        # Data overview
        report_content.append(f"\nDATA OVERVIEW:")
        report_content.append(f"Total records in file: {results.get('total_records', 'N/A')}")
        if 'wai_stats' in results:
            wai_stats = results['wai_stats']
            report_content.append(f"WAI Score Statistics (n={wai_stats['n_wai']}):")
            report_content.append(f"  Mean: {wai_stats['wai_mean']:.2f}")
            report_content.append(f"  Median: {wai_stats['wai_median']:.2f}")
            report_content.append(f"  Std Dev: {wai_stats['wai_std']:.2f}")
            report_content.append(f"  Range: {wai_stats['wai_min']:.1f} - {wai_stats['wai_max']:.1f}")
        else:
            report_content.append("WAI statistics not available.")

        # Requirement 1
        report_content.append(f"\n1. MULTIPLE REGRESSION ANALYSIS:")
        if 'requirement_1' in results:
            r1 = results['requirement_1']
            if isinstance(r1, str):
                report_content.append(f"  Result: {r1}")
            else:
                report_content.append(f"  Sample size: {r1['n_samples']}")
                report_content.append(f"  R² Score: {r1['r2_score']:.4f}")
                report_content.append(f"  Model: WAI = {r1['intercept']:.4f} + {r1['coefficients']['gad_7']:.4f}×GAD-7 + {r1['coefficients']['social_support_scale']:.4f}×Social Support + {r1['coefficients']['brs_scale']:.4f}×BRS")
                report_content.append(f"  Individual correlations:")
                for var, stats in r1['individual_correlations'].items():
                    sig = "***" if stats['p_value'] < 0.001 else "**" if stats['p_value'] < 0.01 else "*" if stats['p_value'] < 0.05 else ""
                    report_content.append(f"    {var.upper()}: r={stats['correlation']:.4f}, p={stats['p_value']:.4f} {sig}")
        else:
            report_content.append("  Analysis not performed or results missing.")
        
        # Requirement 2
        report_content.append(f"\n2. T-TEST ANALYSIS:")
        if 'requirement_2' in results:
            r2_data = results['requirement_2']
            if isinstance(r2_data, str):
                report_content.append(f"  Result: {r2_data}")
            elif not r2_data:
                report_content.append("  No T-test results available.")
            else:
                for instrument, result in r2_data.items():
                    if isinstance(result, str):
                        report_content.append(f"  {instrument.upper()}: {result}")
                    else:
                        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                        report_content.append(f"  {instrument.upper()}: t={result['t_statistic']:.4f}, p={result['p_value']:.4f} {sig}")
                        report_content.append(f"    High WAI group: mean={result['high_wai_mean']:.2f}, n={result['high_wai_n']}")
                        report_content.append(f"    Low WAI group: mean={result['low_wai_mean']:.2f}, n={result['low_wai_n']}")
        else:
            report_content.append("  Analysis not performed or results missing.")
        
        # Requirement 3
        report_content.append(f"\n3. WAI < 27 AND RETURN TO WORK ANALYSIS:")
        if 'requirement_3' in results:
            r3 = results['requirement_3']
            if isinstance(r3, str):
                report_content.append(f"  Result: {r3}")
            else:
                report_content.append(f"  Total participants with WAI data: {r3['total_participants']}")
                report_content.append(f"  Participants with WAI < 27: {r3['low_wai_count']} ({r3['low_wai_percentage']:.1f}%)")
                report_content.append(f"  Returned to work (among WAI < 27): {r3['rtw_count']} out of {r3['low_wai_with_rtw_data']} ({r3['rtw_percentage']:.1f}%)")
        else:
            report_content.append("  Analysis not performed or results missing.")
        
        # Requirement 4
        report_content.append(f"\n4. DEMOGRAPHIC FACTORS ANALYSIS:")
        if 'requirement_4' in results:
            r4_data = results['requirement_4']
            if isinstance(r4_data, str):
                report_content.append(f"  Result: {r4_data}")
            elif not r4_data:
                report_content.append("  No demographic factor results available.")
            else:
                for factor, result in r4_data.items():
                    report_content.append(f"  {factor.replace('_', ' ').title()}:")
                    if isinstance(result, str):
                        report_content.append(f"    Result: {result}")
                    elif 'correlation' in result:
                        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                        report_content.append(f"    Correlation (r): {result['correlation']:.4f}, P-value: {result['p_value']:.4f} {sig}, N: {result['n_samples']}")
                    elif 't_statistic' in result:
                        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                        report_content.append(f"    T-statistic: {result['t_statistic']:.4f}, P-value: {result['p_value']:.4f} {sig}")
                        if 'male_mean' in result and 'female_mean' in result:
                            report_content.append(f"    Male: mean={result['male_mean']:.2f}, n={result['male_n']}")
                            report_content.append(f"    Female: mean={result['female_mean']:.2f}, n={result['female_n']}")
                    elif 'f_statistic' in result:
                        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                        report_content.append(f"    F-statistic: {result['f_statistic']:.4f}, P-value: {result['p_value']:.4f} {sig}")
                        if 'group_means' in result:
                            for group, mean_val in result['group_means'].items():
                                report_content.append(f"    {group}: mean={mean_val:.2f}, n={result['group_counts'][group]}")
        else:
            report_content.append("  Analysis not performed or results missing.")
        
        # Requirement 5
        report_content.append(f"\n5. PAIN AND BLOOD PRESSURE EFFECTS:")
        if 'requirement_5' in results:
            r5 = results['requirement_5']
            if isinstance(r5, str):
                report_content.append(f"  Result: {r5}")
            elif 'pain' in r5 and isinstance(r5['pain'], dict):
                pain_data = r5['pain']
                sig = "***" if pain_data['p_value'] < 0.001 else "**" if pain_data['p_value'] < 0.01 else "*" if pain_data['p_value'] < 0.05 else ""
                report_content.append(f"  Pain: r={pain_data['correlation']:.4f}, p={pain_data['p_value']:.4f} {sig}, n={pain_data['n_samples']}")
            else:
                report_content.append("  Pain analysis not performed or results missing.")
            
            if 'blood_pressure' in r5 and isinstance(r5['blood_pressure'], dict):
                if not r5['blood_pressure']:
                    report_content.append("  Blood pressure analysis not performed or results missing.")
                else:
                    for bp_type, result in r5['blood_pressure'].items():
                        if isinstance(result, dict):
                            sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                            report_content.append(f"  {bp_type.title()} BP: r={result['correlation']:.4f}, p={result['p_value']:.4f} {sig}, n={result['n_samples']}")
            else:
                report_content.append("  Blood pressure analysis not performed or results missing.")
        else:
            report_content.append("  Analysis not performed or results missing.")
        
        # Requirement 6
        report_content.append(f"\n6. GRIP STRENGTH CORRELATION:")
        if 'requirement_6' in results:
            r6 = results['requirement_6']
            if isinstance(r6, str):
                report_content.append(f"  Result: {r6}")
            elif not r6:
                report_content.append("  No grip strength results available.")
            else:
                for grip_type, result in r6.items():
                    if isinstance(result, dict):
                        sig = "***" if result['p_value'] < 0.001 else "**" if result['p_value'] < 0.01 else "*" if result['p_value'] < 0.05 else ""
                        report_content.append(f"  {grip_type.replace('_', ' ').title()}: r={result['correlation']:.4f}, p={result['p_value']:.4f} {sig}, n={result['n_samples']}")
                        report_content.append(f"    WAI mean: {result['wai_mean']:.2f}, Grip strength mean: {result['grip_mean']:.2f}")
        else:
            report_content.append("  Analysis not performed or results missing.")
        
    report_content.append(f"\n" + "="*80)
    report_content.append("ANALYSIS COMPLETED")
    report_content.append("="*80)
    
    return "\n".join(report_content)

def json_serializable(obj):
    """Convert numpy/pandas objects to JSON serializable types"""
    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()
    if pd.isna(obj):
        return None
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

def clean_for_json(data):
    """Recursively clean dictionary/list data for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_for_json(elem) for elem in data]
    else:
        try:
            return json_serializable(data)
        except TypeError:
            return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/upload', methods=['POST'])
def upload_files():
    results_list = []
    app.logger.info('Received upload request')
        
    if 'files[]' not in request.files:
        flash('No files part in the request', 'error')
        app.logger.warning('No files part in the request')
        return redirect(request.url)
        
    files = request.files.getlist('files[]')
    if not files or all(f.filename == '' for f in files):
        flash('No selected file', 'error')
        app.logger.warning('No selected file')
        return redirect(request.url)
        
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(file_path)
                app.logger.info(f'File saved: {file_path}')

                analyzer = AccurateWAIAnalyzer(file_path)
                if analyzer.run_complete_analysis():
                    # Add metadata to results
                    analyzer.results['file_name'] = filename
                    analyzer.results['total_records'] = len(analyzer.data)
                    analyzer.results['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Add WAI stats to results if available
                    if 'work_ability_index' in analyzer.data.columns:
                        wai_data = analyzer.data['work_ability_index'].dropna()
                        if not wai_data.empty:
                            analyzer.results['wai_stats'] = {
                                'n_wai': len(wai_data),
                                'wai_mean': wai_data.mean(),
                                'wai_median': wai_data.median(),
                                'wai_std': wai_data.std(),
                                'wai_min': wai_data.min(),
                                'wai_max': wai_data.max()
                            }

                    results_list.append(clean_for_json(analyzer.results))
                    app.logger.info(f'Analysis complete for {filename}')
                else:
                    analyzer.results['file_name'] = filename
                    analyzer.results['analysis_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    if 'error' not in analyzer.results:
                        analyzer.results['error'] = "Failed to complete analysis."
                    results_list.append(clean_for_json(analyzer.results))
                    app.logger.error(f'Analysis failed for {filename}')

            except Exception as e:
                app.logger.error(f'Error processing {filename}: {e}')
                app.logger.error(traceback.format_exc())
                results_list.append({
                    'file_name': filename,
                    'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'error': f'An unexpected error occurred: {str(e)}'
                })
        else:
            app.logger.warning(f'File {file.filename} not allowed or empty.')
            results_list.append({
                'file_name': file.filename,
                'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'error': 'File type not allowed or file is empty.'
            })

    if not results_list:
        flash('No valid files were processed.', 'error')
        app.logger.warning('No valid files were processed.')
        return redirect(request.url)

    # Generate text report
    app.logger.info('Generating text report')
    report_content = generate_report(results_list)
    report_filename = f"wai_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    report_path = os.path.join(app.config['REPORTS_FOLDER'], report_filename)
    with open(report_path, 'w') as f:
        f.write(report_content)
    app.logger.info(f'Text report generated: {report_filename}')

    # Store the latest report filename in session for download
    session['latest_report'] = report_filename

    return render_template('results.html', results_list=results_list, report_filename=report_filename)

@app.route('/download')
def download_latest_report():
    latest_report = session.get('latest_report')
    if latest_report:
        report_path = os.path.join(app.config['REPORTS_FOLDER'], latest_report)
        if os.path.exists(report_path):
            try:
                return send_file(report_path, as_attachment=True, download_name=latest_report, mimetype='text/plain')
            except Exception as e:
                app.logger.error(f'Error sending file {latest_report}: {e}')
                flash(f'Error downloading report: {e}', 'error')
                return redirect(url_for('index'))
        else:
            app.logger.error(f'Report file not found: {report_path}')
            flash('Report file not found.', 'error')
    else:
        flash('No report available for download. Please upload a file and run analysis first.', 'warning')
    return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_report(filename):
    report_path = os.path.join(app.config['REPORTS_FOLDER'], filename)
    if os.path.exists(report_path):
        try:
            return send_file(report_path, as_attachment=True, download_name=filename, mimetype='text/plain')
        except Exception as e:
            app.logger.error(f'Error sending file {filename}: {e}')
            flash(f'Error downloading report: {e}', 'error')
            return redirect(url_for('index'))
    else:
        app.logger.error(f'Report file not found: {report_path}')
        flash('Report file not found.', 'error')
        return redirect(url_for('index'))

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True) 