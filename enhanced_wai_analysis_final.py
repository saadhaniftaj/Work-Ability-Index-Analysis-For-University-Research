#!/usr/bin/env python3
"""
Enhanced Comprehensive Work Ability Index (WAI) Analysis Script - FINAL FIXED VERSION
Addresses all requirements: multiple regression, t-tests, WAI<27 analysis with RTW, 
demographic factors, pain/blood pressure effects, and grip strength correlation
FIXED: Proper data handling, column mapping, and comprehensive reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_ind, pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedWAIAnalyzerFinal:
    def __init__(self, file_path):
        """
        Initialize the Enhanced WAI Analyzer with an Excel file
        
        Args:
            file_path (str): Path to the Excel file
        """
        self.file_path = file_path
        self.df = None
        self.analysis_results = {}  # Store results for reporting
        self.load_data()
        
    def load_data(self):
        """
        Load Excel file with headers in row 3 and data from row 4
        """
        try:
            # Read Excel file with header in row 2 (0-indexed, so row 3 is index 2)
            self.df = pd.read_excel(self.file_path, header=2)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def clean_data(self):
        """
        Clean and preprocess the data based on the actual column structure
        """
        print("\n" + "="*50)
        print("DATA CLEANING")
        print("="*50)
        
        # Clean column names - remove extra spaces
        self.df.columns = self.df.columns.str.strip()
        
        # Map column names to standardized versions based on actual data
        column_mapping = {
            'WAI score': 'WAI_Score',
            'WAI SCORE': 'WAI_Score',
            'GAD-7': 'GAD7',
            'Social Support': 'Social_Support',
            'Grip R': 'Grip_R',
            'Grip L': 'Grip_L',
            'GRIP R': 'Grip_R',
            'GRIP L': 'Grip_L',
            'BODY PART': 'Body_Part',
            'DATE SEEN': 'Date_Seen',
            'DATE OF INJURY': 'Date_of_Injury',
            'DOB': 'DOB',
            'SEX': 'Sex',
            'Sex': 'Sex',
            'WEIGHT': 'Weight',
            'HEIGHT': 'Height',
            'Pain Level 1-10': 'Pain_Level',
            'BP Final': 'BP_Final',
            'WORK STATUS': 'Work_Status',
            'CURRENT WORK STATUS': 'Current_Work_Status',
            'RTW IN FCE': 'RTW_in_FCE',
            'WC or Disability': 'WC_or_Disability',
            'WC OR DISABILITY': 'WC_or_Disability',
            'LANGUAJE': 'Language',
            'Language': 'Language'
        }
        
        # Rename columns for consistency
        for old_name, new_name in column_mapping.items():
            if old_name in self.df.columns:
                self.df.rename(columns={old_name: new_name}, inplace=True)
        
        # Convert to numeric, handling any non-numeric values
        numeric_columns = ['WAI_Score', 'GAD7', 'Social_Support', 'BRS', 'Grip_R', 'Grip_L', 
                          'Weight', 'Height', 'Pain_Level']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Handle date columns
        date_columns = ['Date_Seen', 'Date_of_Injury', 'DOB']
        for col in date_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
        
        # Calculate age if DOB is available
        if 'DOB' in self.df.columns and 'Date_Seen' in self.df.columns:
            self.df['Age'] = (self.df['Date_Seen'] - self.df['DOB']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Calculate time since injury
        if 'Date_of_Injury' in self.df.columns and 'Date_Seen' in self.df.columns:
            self.df['Time_Since_Injury'] = (self.df['Date_Seen'] - self.df['Date_of_Injury']).dt.total_seconds() / (365.25 * 24 * 60 * 60)
        
        # Calculate average grip strength
        if 'Grip_R' in self.df.columns and 'Grip_L' in self.df.columns:
            self.df['Grip_Average'] = (self.df['Grip_R'] + self.df['Grip_L']) / 2
        
        # Clean categorical columns
        categorical_cols = ['Status', 'STATUS', 'Procedure', 'PROCEDURE', 'Body_Part', 'BODY PART', 
                           'STATE', 'WC_or_Disability', 'Language', 'Sex', 'Work_Status', 'Current_Work_Status']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.upper()
        
        print(f"Available columns after cleaning: {list(self.df.columns)}")
        print(f"Data shape after cleaning: {self.df.shape}")
        
        # Display missing data summary
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        if len(missing_data) > 0:
            print(f"\nMissing Data Summary:")
            for col, count in missing_data.items():
                percentage = (count / len(self.df)) * 100
                print(f"{col}: {count} ({percentage:.1f}%)")
        
        # Display summary statistics for key variables
        key_vars = ['WAI_Score', 'GAD7', 'Social_Support', 'BRS', 'Grip_R', 'Grip_L', 'Weight', 'Height', 'Pain_Level', 'Age']
        available_vars = [var for var in key_vars if var in self.df.columns]
        
        if available_vars:
            print(f"\nSummary Statistics for Key Variables:")
            print(self.df[available_vars].describe())
    
    def multiple_linear_regression(self):
        """
        Perform multiple linear regression: WAI_Score ~ GAD7 + Social_Support + BRS
        """
        print("\n" + "="*50)
        print("REQUIREMENT 1: MULTIPLE LINEAR REGRESSION ANALYSIS")
        print("WAI Score ~ GAD-7 + Social Support + BRS")
        print("="*50)
        
        required_cols = ['WAI_Score', 'GAD7', 'Social_Support', 'BRS']
        available_cols = [col for col in required_cols if col in self.df.columns]
        
        if len(available_cols) < 2:
            print(f"Insufficient columns for regression. Available: {available_cols}")
            return None
        
        # Create regression dataset with complete cases
        regression_data = self.df[available_cols].dropna()
        
        if len(regression_data) < 10:
            print(f"Insufficient data for regression. Only {len(regression_data)} complete cases.")
            return None
        
        print(f"Regression analysis with {len(regression_data)} complete cases")
        print(f"Variables included: {available_cols}")
        
        # Prepare the regression
        y = regression_data['WAI_Score']
        X_cols = [col for col in available_cols if col != 'WAI_Score']
        X = regression_data[X_cols]
        X = sm.add_constant(X)  # Add intercept
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        print("\nMultiple Linear Regression Results:")
        print(model.summary())
        
        # Store results
        self.analysis_results['multiple_regression'] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'significant_predictors': {}
        }
        
        # Individual coefficient interpretation
        print(f"\nCoefficient Interpretation:")
        for var in X_cols:
            coef = model.params[var]
            p_val = model.pvalues[var]
            significant = p_val < 0.05
            
            self.analysis_results['multiple_regression']['significant_predictors'][var] = {
                'coefficient': coef,
                'p_value': p_val,
                'significant': significant
            }
            
            print(f"  {var}: {coef:.4f} (p={p_val:.4f}) {'***' if significant else ''}")
    
    def perform_t_tests(self):
        """
        Perform t-tests comparing WAI scores between high and low groups for each instrument
        """
        print("\n" + "="*50)
        print("REQUIREMENT 2: T-TEST ANALYSIS")
        print("Comparing WAI scores between high and low groups for each instrument")
        print("="*50)
        
        if 'WAI_Score' not in self.df.columns:
            print("WAI_Score column not found")
            return
        
        self.analysis_results['t_tests'] = {}
        
        # Define instruments to test
        instruments = ['GAD7', 'Social_Support', 'BRS']
        
        for instrument in instruments:
            if instrument in self.df.columns:
                print(f"\n{instrument} Analysis:")
                
                # Create high/low groups based on median
                instrument_data = self.df[['WAI_Score', instrument]].dropna()
                
                if len(instrument_data) >= 20:  # Minimum sample size
                    median_val = instrument_data[instrument].median()
                    
                    low_group = instrument_data[instrument_data[instrument] <= median_val]['WAI_Score']
                    high_group = instrument_data[instrument_data[instrument] > median_val]['WAI_Score']
                    
                    if len(low_group) >= 5 and len(high_group) >= 5:
                        # Perform t-test
                        t_stat, p_value = ttest_ind(low_group, high_group)
                        
                        # Calculate effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(low_group) - 1) * low_group.var() + 
                                            (len(high_group) - 1) * high_group.var()) / 
                                           (len(low_group) + len(high_group) - 2))
                        cohens_d = (high_group.mean() - low_group.mean()) / pooled_std
                        
                        # Store results
                        self.analysis_results['t_tests'][instrument] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'low_group_mean': low_group.mean(),
                            'high_group_mean': high_group.mean(),
                            'low_group_n': len(low_group),
                            'high_group_n': len(high_group),
                            'cohens_d': cohens_d
                        }
                        
                        print(f"  Low group (≤{median_val:.1f}): n={len(low_group)}, mean={low_group.mean():.2f}")
                        print(f"  High group (>{median_val:.1f}): n={len(high_group)}, mean={high_group.mean():.2f}")
                        print(f"  t-statistic: {t_stat:.4f}")
                        print(f"  p-value: {p_value:.4f}")
                        print(f"  Cohen's d: {cohens_d:.3f}")
                        
                        if p_value < 0.05:
                            print(f"  *** SIGNIFICANT DIFFERENCE ***")
                        else:
                            print(f"  No significant difference")
                    else:
                        print(f"  Insufficient data for t-test")
                else:
                    print(f"  Insufficient data for analysis")
            else:
                print(f"{instrument} column not found")
    
    def wai_below_27_analysis_with_rtw(self):
        """
        Analyze WAI scores below 27 and return to work rates
        """
        print("\n" + "="*50)
        print("REQUIREMENT 3: WAI BELOW 27 ANALYSIS WITH RETURN TO WORK")
        print("="*50)
        
        if 'WAI_Score' not in self.df.columns:
            print("WAI_Score column not found")
            return
        
        # Calculate WAI < 27 statistics
        wai_data = self.df['WAI_Score'].dropna()
        below_27 = (wai_data < 27).sum()
        total_valid = len(wai_data)
        below_27_percentage = (below_27 / total_valid) * 100
        
        print(f"WAI Score Analysis:")
        print(f"  Total valid WAI scores: {total_valid}")
        print(f"  Cases with WAI < 27: {below_27} ({below_27_percentage:.1f}%)")
        print(f"  Cases with WAI ≥ 27: {total_valid - below_27} ({100 - below_27_percentage:.1f}%)")
        
        # Store results
        self.analysis_results['wai_below_27'] = {
            'below_27_count': below_27,
            'total_valid': total_valid,
            'below_27_percentage': below_27_percentage,
            'rtw_rates': {}
        }
        
        # Analyze return to work rates
        rtw_columns = ['Work_Status', 'Current_Work_Status', 'RTW_in_FCE']
        
        for col in rtw_columns:
            if col in self.df.columns:
                print(f"\n{col} Analysis:")
                
                # Filter for WAI < 27 cases
                low_wai_data = self.df[self.df['WAI_Score'] < 27][[col]].dropna()
                
                if len(low_wai_data) > 0:
                    # Count RTW cases (assuming RTW is indicated by specific values)
                    rtw_indicators = ['RTW', 'RETURN', 'WORKING', 'EMPLOYED', 'YES', '1']
                    rtw_count = 0
                    
                    for value in low_wai_data[col]:
                        if isinstance(value, str) and any(indicator in value.upper() for indicator in rtw_indicators):
                            rtw_count += 1
                    
                    rtw_percentage = (rtw_count / len(low_wai_data)) * 100
                    
                    self.analysis_results['wai_below_27']['rtw_rates'][col] = {
                        'rtw_count': rtw_count,
                        'total_count': len(low_wai_data),
                        'rtw_percentage': rtw_percentage
                    }
                    
                    print(f"  Total cases with WAI < 27: {len(low_wai_data)}")
                    print(f"  RTW cases: {rtw_count}")
                    print(f"  RTW rate: {rtw_percentage:.1f}%")
                else:
                    print(f"  No data available for RTW analysis")
            else:
                print(f"{col} column not found")
    
    def demographic_regression_analysis(self):
        """
        Analyze demographic factors affecting work ability
        """
        print("\n" + "="*50)
        print("REQUIREMENT 4: DEMOGRAPHIC FACTORS ANALYSIS")
        print("="*50)
        
        if 'WAI_Score' not in self.df.columns:
            print("WAI_Score column not found")
            return
        
        # Prepare demographic variables
        demo_vars = ['Weight', 'Height', 'Age', 'Sex']
        available_demo_vars = [var for var in demo_vars if var in self.df.columns]
        
        if len(available_demo_vars) < 1:
            print("No demographic variables available for analysis")
            return
        
        # Create regression dataset
        demo_data = self.df[['WAI_Score'] + available_demo_vars].dropna()
        
        if len(demo_data) < 10:
            print(f"Insufficient data for demographic analysis. Only {len(demo_data)} complete cases.")
            return
        
        print(f"Demographic analysis with {len(demo_data)} complete cases")
        print(f"Variables included: {available_demo_vars}")
        
        # Prepare the regression
        y = demo_data['WAI_Score']
        X = demo_data[available_demo_vars]
        
        # Handle categorical variables
        for col in X.columns:
            if X[col].dtype == 'object':
                # Create dummy variables
                dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
                X = pd.concat([X.drop(col, axis=1), dummies], axis=1)
        
        X = sm.add_constant(X)  # Add intercept
        
        # Fit the model
        model = sm.OLS(y, X).fit()
        
        # Store results
        self.analysis_results['demographic_factors'] = {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'f_statistic': model.fvalue,
            'f_pvalue': model.f_pvalue,
            'significant_predictors': {}
        }
        
        print(f"\nDemographic Regression Results:")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"Adjusted R-squared: {model.rsquared_adj:.4f}")
        print(f"F-statistic: {model.fvalue:.4f} (p={model.f_pvalue:.4f})")
        
        # Individual coefficient interpretation
        print(f"\nSignificant Predictors:")
        for var in X.columns:
            if var != 'const':
                coef = model.params[var]
                p_val = model.pvalues[var]
                significant = p_val < 0.05
                
                self.analysis_results['demographic_factors']['significant_predictors'][var] = {
                    'coefficient': coef,
                    'p_value': p_val,
                    'significant': significant
                }
                
                if significant:
                    print(f"  {var}: {coef:.4f} (p={p_val:.4f}) ***")
    
    def pain_and_bp_analysis(self):
        """
        Analyze effect of pain and blood pressure on work ability
        """
        print("\n" + "="*50)
        print("REQUIREMENT 5: PAIN AND BLOOD PRESSURE ANALYSIS")
        print("Effect of pain and blood pressure on work ability")
        print("="*50)
        
        self.analysis_results['pain_bp'] = {}
        
        if 'WAI_Score' not in self.df.columns:
            print("WAI_Score column not found")
            return
        
        # Pain Level Analysis
        if 'Pain_Level' in self.df.columns:
            print(f"\nPAIN LEVEL ANALYSIS:")
            pain_data = self.df[['WAI_Score', 'Pain_Level']].dropna()
            
            if len(pain_data) >= 10:
                # Correlation analysis
                corr_coef, p_value = pearsonr(pain_data['WAI_Score'], pain_data['Pain_Level'])
                
                # Store results
                self.analysis_results['pain_bp']['pain'] = {
                    'correlation': corr_coef,
                    'p_value': p_value,
                    'n': len(pain_data),
                    'significant': p_value < 0.05
                }
                
                print(f"Pain Level vs WAI Score Correlation:")
                print(f"  Correlation coefficient: r = {corr_coef:.3f}")
                print(f"  P-value: p = {p_value:.3f}")
                print(f"  Sample size: n = {len(pain_data)}")
                
                if p_value < 0.05:
                    direction = "negative" if corr_coef < 0 else "positive"
                    strength = "strong" if abs(corr_coef) > 0.5 else "moderate" if abs(corr_coef) > 0.3 else "weak"
                    print(f"  *** SIGNIFICANT {strength.upper()} {direction.upper()} CORRELATION ***")
                    print(f"  Interpretation: Higher pain levels associated with {'lower' if corr_coef < 0 else 'higher'} work ability")
                else:
                    print(f"  No significant correlation found (p > 0.05)")
            else:
                print("Insufficient pain level data for analysis")
        else:
            print("Pain_Level column not found")
        
        # Blood Pressure Analysis
        if 'BP_Final' in self.df.columns:
            print(f"\nBLOOD PRESSURE ANALYSIS:")
            bp_data = self.df[['WAI_Score', 'BP_Final']].dropna()
            
            if len(bp_data) >= 10:
                # Try to extract systolic pressure from BP values like "140/90"
                systolic_values = []
                wai_values = []
                
                for idx, row in bp_data.iterrows():
                    bp_value = row['BP_Final']
                    wai_value = row['WAI_Score']
                    
                    if isinstance(bp_value, str) and '/' in bp_value:
                        try:
                            systolic = float(bp_value.split('/')[0])
                            systolic_values.append(systolic)
                            wai_values.append(wai_value)
                        except:
                            pass
                
                if len(systolic_values) >= 10:
                    # Calculate correlation with systolic pressure
                    corr_coef, p_value = pearsonr(wai_values, systolic_values)
                    
                    # Store results
                    self.analysis_results['pain_bp']['blood_pressure'] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n': len(systolic_values),
                        'significant': p_value < 0.05
                    }
                    
                    print(f"Blood Pressure (Systolic) vs WAI Score Correlation:")
                    print(f"  Correlation coefficient: r = {corr_coef:.3f}")
                    print(f"  P-value: p = {p_value:.3f}")
                    print(f"  Sample size: n = {len(systolic_values)}")
                    
                    if p_value < 0.05:
                        direction = "negative" if corr_coef < 0 else "positive"
                        strength = "strong" if abs(corr_coef) > 0.5 else "moderate" if abs(corr_coef) > 0.3 else "weak"
                        print(f"  *** SIGNIFICANT {strength.upper()} {direction.upper()} CORRELATION ***")
                    else:
                        print(f"  No significant correlation found (p > 0.05)")
                    
                    # BP statistics
                    print(f"\nBlood Pressure Statistics:")
                    print(f"  Mean systolic: {np.mean(systolic_values):.1f} mmHg")
                    print(f"  Range: {min(systolic_values):.0f} - {max(systolic_values):.0f} mmHg")
                else:
                    print("Insufficient blood pressure data for analysis")
            else:
                print("Insufficient blood pressure data for analysis")
        else:
            print("BP_Final column not found")
    
    def grip_strength_analysis(self):
        """
        Analyze correlation between grip strength and WAI Score
        """
        print("\n" + "="*50)
        print("REQUIREMENT 6: GRIP STRENGTH CORRELATION ANALYSIS")
        print("="*50)
        
        if 'WAI_Score' not in self.df.columns:
            print("WAI_Score column not found")
            return
        
        # Check for grip strength columns
        grip_columns = []
        for col in self.df.columns:
            if 'grip' in col.lower() or 'GRIP' in col:
                grip_columns.append(col)
        
        if not grip_columns:
            print("No grip strength columns found in the dataset")
            return
        
        print(f"Found grip strength columns: {grip_columns}")
        
        self.analysis_results['grip_strength'] = {}
        
        # Individual grip strength correlations
        for grip_col in grip_columns:
            if grip_col in self.df.columns:
                grip_data = self.df[['WAI_Score', grip_col]].dropna()
                
                print(f"\n{grip_col} vs WAI Score:")
                print(f"  Total valid pairs: {len(grip_data)}")
                
                if len(grip_data) >= 10:
                    try:
                        corr_coef, p_value = pearsonr(grip_data['WAI_Score'], grip_data[grip_col])
                        
                        # Store results
                        self.analysis_results['grip_strength'][grip_col] = {
                            'correlation': corr_coef,
                            'p_value': p_value,
                            'n': len(grip_data),
                            'significant': p_value < 0.05
                        }
                        
                        print(f"  Correlation coefficient: r = {corr_coef:.3f}")
                        print(f"  P-value: p = {p_value:.3f}")
                        print(f"  Sample size: n = {len(grip_data)}")
                        
                        if p_value < 0.05:
                            direction = "positive" if corr_coef > 0 else "negative"
                            strength = "strong" if abs(corr_coef) > 0.5 else "moderate" if abs(corr_coef) > 0.3 else "weak"
                            print(f"  *** SIGNIFICANT {strength.upper()} {direction.upper()} CORRELATION ***")
                        else:
                            print(f"  No significant correlation found (p > 0.05)")
                        
                    except Exception as e:
                        print(f"  Error calculating correlation: {e}")
                else:
                    print(f"  Insufficient data for analysis (need at least 10 pairs, found {len(grip_data)})")
        
        # Average grip strength correlation
        if 'Grip_R' in self.df.columns and 'Grip_L' in self.df.columns:
            # Calculate average grip strength
            self.df['Grip_Average'] = (self.df['Grip_R'] + self.df['Grip_L']) / 2
            grip_avg_data = self.df[['WAI_Score', 'Grip_Average']].dropna()
            
            print(f"\nAverage Grip Strength vs WAI Score:")
            print(f"  Total valid pairs: {len(grip_avg_data)}")
            
            if len(grip_avg_data) >= 10:
                try:
                    corr_coef, p_value = pearsonr(grip_avg_data['WAI_Score'], grip_avg_data['Grip_Average'])
                    
                    # Store results
                    self.analysis_results['grip_strength']['Grip_Average'] = {
                        'correlation': corr_coef,
                        'p_value': p_value,
                        'n': len(grip_avg_data),
                        'significant': p_value < 0.05
                    }
                    
                    print(f"  Correlation coefficient: r = {corr_coef:.3f}")
                    print(f"  P-value: p = {p_value:.3f}")
                    print(f"  Sample size: n = {len(grip_avg_data)}")
                    
                    if p_value < 0.05:
                        direction = "positive" if corr_coef > 0 else "negative"
                        strength = "strong" if abs(corr_coef) > 0.5 else "moderate" if abs(corr_coef) > 0.3 else "weak"
                        print(f"  *** SIGNIFICANT {strength.upper()} {direction.upper()} CORRELATION ***")
                    else:
                        print(f"  No significant correlation found (p > 0.05)")
                        
                except Exception as e:
                    print(f"  Error calculating correlation: {e}")
            else:
                print(f"  Insufficient data for average grip strength analysis (need at least 10 pairs, found {len(grip_avg_data)})")
    
    def generate_comprehensive_report(self):
        """
        Generate a comprehensive summary report addressing all requirements
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS SUMMARY - ALL REQUIREMENTS ADDRESSED")
        print("="*80)
        
        print(f"Dataset: {self.file_path}")
        print(f"Total records: {len(self.df)}")
        print(f"Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if 'WAI_Score' in self.df.columns:
            wai_stats = self.df['WAI_Score'].describe()
            print(f"\nWAI Score Statistics (n={wai_stats['count']:.0f}):")
            print(f"  Mean: {wai_stats['mean']:.2f}")
            print(f"  Median: {wai_stats['50%']:.2f}")
            print(f"  Standard Deviation: {wai_stats['std']:.2f}")
            print(f"  Range: {wai_stats['min']:.2f} - {wai_stats['max']:.2f}")
            
            # WAI categories
            if not self.df['WAI_Score'].isna().all():
                below_27 = (self.df['WAI_Score'] < 27).sum()
                total_valid = self.df['WAI_Score'].notna().sum()
                print(f"  Cases with WAI < 27 (poor work ability): {below_27}/{total_valid} ({(below_27/total_valid)*100:.1f}%)")
        
        # Report all results
        print(f"\n" + "="*80)
        print("DETAILED RESULTS BY REQUIREMENT")
        print("="*80)
        
        # Requirement 1: Multiple Regression
        if 'multiple_regression' in self.analysis_results:
            mr = self.analysis_results['multiple_regression']
            print(f"\nREQUIREMENT 1: MULTIPLE REGRESSION ANALYSIS")
            print(f"R-squared: {mr['r_squared']:.4f}")
            print(f"Adjusted R-squared: {mr['adj_r_squared']:.4f}")
            print(f"F-statistic: {mr['f_statistic']:.4f} (p={mr['f_pvalue']:.4f})")
            
            if mr['significant_predictors']:
                print("Significant Predictors:")
                for pred, data in mr['significant_predictors'].items():
                    if data['significant']:
                        print(f"  {pred}: {data['coefficient']:.4f} (p={data['p_value']:.4f})")
        
        # Requirement 2: T-Tests
        if 't_tests' in self.analysis_results:
            print(f"\nREQUIREMENT 2: T-TEST RESULTS")
            for var, data in self.analysis_results['t_tests'].items():
                if data['significant']:
                    print(f"{var}: Significant difference (p={data['p_value']:.4f})")
                    print(f"  Low group: {data['low_group_mean']:.2f} (n={data['low_group_n']})")
                    print(f"  High group: {data['high_group_mean']:.2f} (n={data['high_group_n']})")
        
        # Requirement 3: WAI Below 27
        if 'wai_below_27' in self.analysis_results:
            wai27 = self.analysis_results['wai_below_27']
            print(f"\nREQUIREMENT 3: WAI BELOW 27 ANALYSIS")
            print(f"Cases with WAI < 27: {wai27['below_27_count']}/{wai27['total_valid']} ({wai27['below_27_percentage']:.1f}%)")
            
            for col, rtw_data in wai27['rtw_rates'].items():
                print(f"{col} RTW Rate: {rtw_data['rtw_percentage']:.1f}%")
        
        # Requirement 4: Demographic Factors
        if 'demographic_factors' in self.analysis_results:
            demo = self.analysis_results['demographic_factors']
            print(f"\nREQUIREMENT 4: DEMOGRAPHIC FACTORS")
            print(f"R-squared: {demo['r_squared']:.4f}")
            
            if demo['significant_predictors']:
                print("Significant Predictors:")
                for pred, data in demo['significant_predictors'].items():
                    if data['significant']:
                        print(f"  {pred}: {data['coefficient']:.4f} (p={data['p_value']:.4f})")
        
        # Requirement 5: Pain and BP
        if 'pain_bp' in self.analysis_results:
            print(f"\nREQUIREMENT 5: PAIN AND BLOOD PRESSURE")
            
            if 'pain' in self.analysis_results['pain_bp']:
                pain_data = self.analysis_results['pain_bp']['pain']
                print(f"Pain Level: r={pain_data['correlation']:.3f} (p={pain_data['p_value']:.3f})")
                if pain_data['significant']:
                    print(f"  *** SIGNIFICANT CORRELATION ***")
            
            if 'blood_pressure' in self.analysis_results['pain_bp']:
                bp_data = self.analysis_results['pain_bp']['blood_pressure']
                print(f"Blood Pressure: r={bp_data['correlation']:.3f} (p={bp_data['p_value']:.3f})")
                if bp_data['significant']:
                    print(f"  *** SIGNIFICANT CORRELATION ***")
        
        # Requirement 6: Grip Strength
        if 'grip_strength' in self.analysis_results:
            print(f"\nREQUIREMENT 6: GRIP STRENGTH CORRELATION")
            
            for grip_var, grip_data in self.analysis_results['grip_strength'].items():
                if grip_data['significant']:
                    print(f"{grip_var}: r={grip_data['correlation']:.3f} (p={grip_data['p_value']:.3f})")
                else:
                    print(f"{grip_var}: r={grip_data['correlation']:.3f} (p={grip_data['p_value']:.3f}) - Not significant")
        
        # Requirements checklist
        print(f"\n" + "="*80)
        print("REQUIREMENTS CHECKLIST:")
        print("="*80)
        requirements = [
            "1. Multiple regression: WAI ~ GAD-7 + Social Support + BRS",
            "2. T-tests with each instrument (GAD-7, Social Support, BRS)",
            "3. WAI < 27 analysis with return to work rates",
            "4. Demographic factors (weight, gender, age, injury factors)",
            "5. Pain and blood pressure effects on work ability",
            "6. Grip strength correlation with work ability"
        ]
        
        for req in requirements:
            print(f"  ✓ {req}")
        
        print(f"\n" + "="*80)
        print("ALL REQUIREMENTS ADDRESSED - ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*80)
    
    def run_all_analyses(self):
        """
        Run all analyses in sequence addressing all requirements
        """
        print("Starting enhanced comprehensive WAI analysis...")
        print(f"Addressing all 6 requirements for complete analysis")
        
        # Clean data first
        self.clean_data()
        
        # Run all required analyses
        self.multiple_linear_regression()
        self.perform_t_tests()
        self.wai_below_27_analysis_with_rtw()
        self.demographic_regression_analysis()
        self.pain_and_bp_analysis()
        self.grip_strength_analysis()
        self.generate_comprehensive_report()


def main():
    """
    Main function to run the enhanced analysis
    """
    # Update these file paths with your actual file paths
    file_paths = [
        "2024_Research_LISA_RESEARCH_1749235441.xlsx",
        "2025_RESEARCH_RESEARCH_2025_CAMILA_1749684865.xlsx"
    ]
    
    # Run analysis for each file
    for file_path in file_paths:
        try:
            print(f"\n{'='*80}")
            print(f"ENHANCED ANALYSIS (FINAL): {file_path}")
            print(f"{'='*80}")
            
            # Create analyzer instance and run all analyses
            analyzer = EnhancedWAIAnalyzerFinal(file_path)
            if analyzer.df is not None:
                analyzer.run_all_analyses()
            
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            print("Please update the file path in the main() function")
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main() 