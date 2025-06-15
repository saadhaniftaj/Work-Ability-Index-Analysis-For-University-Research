#!/usr/bin/env python3
"""
WAI Analysis Validation Module
Comprehensive validation methods for Work Ability Index analysis results
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, jarque_bera, levene, bartlett, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class WAIAnalysisValidator:
    """
    Comprehensive validation class for WAI analysis results
    """
    
    def __init__(self, df, analysis_results=None):
        """
        Initialize validator with data and optional analysis results
        
        Args:
            df (pd.DataFrame): The dataset used for analysis
            analysis_results (dict): Optional dictionary containing analysis results
        """
        self.df = df.copy()
        self.analysis_results = analysis_results or {}
        self.validation_results = {}
        
    def validate_data_quality(self):
        """
        Comprehensive data quality validation
        """
        print("\n" + "="*60)
        print("DATA QUALITY VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        # 1. Basic data structure validation
        print("1. Data Structure Validation:")
        validation_results['data_structure'] = {
            'total_records': len(self.df),
            'total_columns': len(self.df.columns),
            'missing_values_total': self.df.isnull().sum().sum(),
            'duplicate_records': self.df.duplicated().sum()
        }
        
        print(f"   Total records: {validation_results['data_structure']['total_records']}")
        print(f"   Total columns: {validation_results['data_structure']['total_columns']}")
        print(f"   Total missing values: {validation_results['data_structure']['missing_values_total']}")
        print(f"   Duplicate records: {validation_results['data_structure']['duplicate_records']}")
        
        # 2. Missing data analysis
        print("\n2. Missing Data Analysis:")
        missing_data = self.df.isnull().sum()
        missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
        
        validation_results['missing_data'] = {}
        for col, count in missing_data.items():
            percentage = (count / len(self.df)) * 100
            validation_results['missing_data'][col] = {
                'count': count,
                'percentage': percentage,
                'severity': 'high' if percentage > 20 else 'moderate' if percentage > 5 else 'low'
            }
            print(f"   {col}: {count} ({percentage:.1f}%) - {validation_results['missing_data'][col]['severity']} severity")
        
        # 3. Data type validation
        print("\n3. Data Type Validation:")
        validation_results['data_types'] = {}
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            validation_results['data_types'][col] = dtype
            print(f"   {col}: {dtype}")
        
        # 4. Outlier detection for numeric columns
        print("\n4. Outlier Detection:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        validation_results['outliers'] = {}
        
        for col in numeric_cols:
            if col in self.df.columns and not self.df[col].isna().all():
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                outlier_percentage = (outliers / self.df[col].notna().sum()) * 100
                
                validation_results['outliers'][col] = {
                    'count': outliers,
                    'percentage': outlier_percentage,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
                
                print(f"   {col}: {outliers} outliers ({outlier_percentage:.1f}%)")
        
        # 5. Range validation for key variables
        print("\n5. Range Validation for Key Variables:")
        key_vars = {
            'WAI_Score': (0, 49),
            'GAD7': (0, 21),
            'Social_Support': (0, 100),
            'BRS': (0, 100),
            'Pain_Level': (0, 10),
            'Grip_R': (0, 100),
            'Grip_L': (0, 100)
        }
        
        validation_results['range_validation'] = {}
        for var, (min_val, max_val) in key_vars.items():
            if var in self.df.columns:
                actual_min = self.df[var].min()
                actual_max = self.df[var].max()
                out_of_range = ((self.df[var] < min_val) | (self.df[var] > max_val)).sum()
                
                validation_results['range_validation'][var] = {
                    'expected_range': (min_val, max_val),
                    'actual_range': (actual_min, actual_max),
                    'out_of_range_count': out_of_range,
                    'is_valid': actual_min >= min_val and actual_max <= max_val
                }
                
                status = "✓ VALID" if validation_results['range_validation'][var]['is_valid'] else "✗ INVALID"
                print(f"   {var}: {status} - Range: {actual_min:.1f}-{actual_max:.1f} (expected: {min_val}-{max_val})")
        
        self.validation_results['data_quality'] = validation_results
        return validation_results
    
    def validate_statistical_assumptions(self, regression_data=None):
        """
        Validate statistical assumptions for regression analysis
        """
        print("\n" + "="*60)
        print("STATISTICAL ASSUMPTIONS VALIDATION")
        print("="*60)
        
        if regression_data is None:
            # Use WAI regression data if available
            required_cols = ['WAI_Score', 'GAD7', 'Social_Support', 'BRS']
            available_cols = [col for col in required_cols if col in self.df.columns]
            if len(available_cols) >= 2:
                regression_data = self.df[available_cols].dropna()
            else:
                print("Insufficient data for regression assumption validation")
                return None
        
        validation_results = {}
        
        # 1. Normality test for residuals
        print("1. Normality of Residuals:")
        try:
            y = regression_data['WAI_Score']
            X_cols = [col for col in regression_data.columns if col != 'WAI_Score']
            X = regression_data[X_cols]
            X = sm.add_constant(X)
            
            model = sm.OLS(y, X).fit()
            residuals = model.resid
            
            # Shapiro-Wilk test
            shapiro_stat, shapiro_p = shapiro(residuals)
            # Jarque-Bera test
            jb_stat, jb_p = jarque_bera(residuals)
            
            validation_results['normality'] = {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'jarque_bera_statistic': jb_stat,
                'jarque_bera_p_value': jb_p,
                'is_normal': shapiro_p > 0.05 and jb_p > 0.05
            }
            
            print(f"   Shapiro-Wilk test: statistic={shapiro_stat:.4f}, p={shapiro_p:.4f}")
            print(f"   Jarque-Bera test: statistic={jb_stat:.4f}, p={jb_p:.4f}")
            print(f"   Normality assumption: {'✓ MET' if validation_results['normality']['is_normal'] else '✗ VIOLATED'}")
            
        except Exception as e:
            print(f"   Error in normality test: {e}")
        
        # 2. Homoscedasticity test
        print("\n2. Homoscedasticity:")
        try:
            # Breusch-Pagan test
            bp_stat, bp_p, bp_f, bp_f_p = het_breuschpagan(residuals, X)
            # White test
            white_stat, white_p, white_f, white_f_p = het_white(residuals, X)
            
            validation_results['homoscedasticity'] = {
                'breusch_pagan_statistic': bp_stat,
                'breusch_pagan_p_value': bp_p,
                'white_statistic': white_stat,
                'white_p_value': white_p,
                'is_homoscedastic': bp_p > 0.05 and white_p > 0.05
            }
            
            print(f"   Breusch-Pagan test: statistic={bp_stat:.4f}, p={bp_p:.4f}")
            print(f"   White test: statistic={white_stat:.4f}, p={white_p:.4f}")
            print(f"   Homoscedasticity assumption: {'✓ MET' if validation_results['homoscedasticity']['is_homoscedastic'] else '✗ VIOLATED'}")
            
        except Exception as e:
            print(f"   Error in homoscedasticity test: {e}")
        
        # 3. Multicollinearity test
        print("\n3. Multicollinearity:")
        try:
            # Calculate VIF for each predictor
            vif_data = []
            for i, col in enumerate(X_cols):
                vif = variance_inflation_factor(X.values, i + 1)  # +1 for constant
                vif_data.append({'variable': col, 'vif': vif})
            
            validation_results['multicollinearity'] = {
                'vif_values': vif_data,
                'max_vif': max([item['vif'] for item in vif_data]),
                'high_vif_variables': [item['variable'] for item in vif_data if item['vif'] > 10]
            }
            
            print(f"   VIF values:")
            for item in vif_data:
                severity = "HIGH" if item['vif'] > 10 else "MODERATE" if item['vif'] > 5 else "LOW"
                print(f"     {item['variable']}: {item['vif']:.2f} ({severity})")
            
            print(f"   Maximum VIF: {validation_results['multicollinearity']['max_vif']:.2f}")
            print(f"   Multicollinearity: {'✗ PRESENT' if validation_results['multicollinearity']['max_vif'] > 10 else '✓ ACCEPTABLE'}")
            
        except Exception as e:
            print(f"   Error in multicollinearity test: {e}")
        
        # 4. Independence of residuals
        print("\n4. Independence of Residuals:")
        try:
            # Durbin-Watson test
            dw_stat = sm.stats.durbin_watson(residuals)
            
            validation_results['independence'] = {
                'durbin_watson_statistic': dw_stat,
                'is_independent': 1.5 < dw_stat < 2.5
            }
            
            print(f"   Durbin-Watson statistic: {dw_stat:.4f}")
            print(f"   Independence assumption: {'✓ MET' if validation_results['independence']['is_independent'] else '✗ VIOLATED'}")
            
        except Exception as e:
            print(f"   Error in independence test: {e}")
        
        self.validation_results['statistical_assumptions'] = validation_results
        return validation_results
    
    def validate_correlation_results(self, correlation_results):
        """
        Validate correlation analysis results
        """
        print("\n" + "="*60)
        print("CORRELATION ANALYSIS VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        for var_name, corr_data in correlation_results.items():
            print(f"\n{var_name}:")
            
            # Check sample size adequacy
            n = corr_data.get('n', 0)
            min_sample_size = 30  # Minimum for reliable correlation
            
            # Check correlation coefficient validity
            corr_coef = corr_data.get('correlation', 0)
            p_value = corr_data.get('p_value', 1)
            
            # Calculate confidence interval
            if n > 3:
                z = np.arctanh(corr_coef)
                se = 1 / np.sqrt(n - 3)
                ci_lower = np.tanh(z - 1.96 * se)
                ci_upper = np.tanh(z + 1.96 * se)
            else:
                ci_lower = ci_upper = np.nan
            
            validation_results[var_name] = {
                'sample_size': n,
                'sample_size_adequate': n >= min_sample_size,
                'correlation_coefficient': corr_coef,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'confidence_interval': (ci_lower, ci_upper),
                'effect_size': abs(corr_coef),
                'effect_size_interpretation': self._interpret_effect_size(abs(corr_coef))
            }
            
            print(f"   Sample size: {n} {'✓' if n >= min_sample_size else '✗'}")
            print(f"   Correlation: {corr_coef:.4f}")
            print(f"   P-value: {p_value:.4f} {'✓' if p_value < 0.05 else '✗'}")
            print(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"   Effect size: {validation_results[var_name]['effect_size_interpretation']}")
        
        self.validation_results['correlation_validation'] = validation_results
        return validation_results
    
    def validate_t_test_results(self, t_test_results):
        """
        Validate t-test results
        """
        print("\n" + "="*60)
        print("T-TEST VALIDATION")
        print("="*60)
        
        validation_results = {}
        
        for test_name, test_data in t_test_results.items():
            print(f"\n{test_name}:")
            
            # Extract test statistics
            t_stat = test_data.get('t_statistic', 0)
            p_value = test_data.get('p_value', 1)
            n1 = test_data.get('n1', 0)
            n2 = test_data.get('n2', 0)
            df = test_data.get('degrees_of_freedom', 0)
            
            # Calculate effect size (Cohen's d)
            if 'mean_diff' in test_data and 'pooled_std' in test_data:
                cohens_d = test_data['mean_diff'] / test_data['pooled_std']
            else:
                cohens_d = np.nan
            
            validation_results[test_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'sample_sizes': (n1, n2),
                'total_sample_size': n1 + n2,
                'degrees_of_freedom': df,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_cohens_d(abs(cohens_d)) if not np.isnan(cohens_d) else 'Unknown'
            }
            
            print(f"   T-statistic: {t_stat:.4f}")
            print(f"   P-value: {p_value:.4f} {'✓' if p_value < 0.05 else '✗'}")
            print(f"   Sample sizes: {n1}, {n2} (total: {n1 + n2})")
            print(f"   Degrees of freedom: {df}")
            if not np.isnan(cohens_d):
                print(f"   Cohen's d: {cohens_d:.4f} ({validation_results[test_name]['effect_size_interpretation']})")
        
        self.validation_results['t_test_validation'] = validation_results
        return validation_results
    
    def _interpret_effect_size(self, r):
        """
        Interpret correlation effect size
        """
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        else:
            return "large"
    
    def _interpret_cohens_d(self, d):
        """
        Interpret Cohen's d effect size
        """
        if d < 0.2:
            return "small"
        elif d < 0.5:
            return "medium"
        elif d < 0.8:
            return "large"
        else:
            return "very large"
    
    def generate_validation_report(self):
        """
        Generate comprehensive validation report
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE VALIDATION REPORT")
        print("="*80)
        
        # Overall validation summary
        print("VALIDATION SUMMARY:")
        
        # Data quality summary
        if 'data_quality' in self.validation_results:
            dq = self.validation_results['data_quality']
            print(f"  Data Quality: {'✓ GOOD' if dq['data_structure']['missing_values_total'] < len(self.df) * 0.1 else '⚠ CONCERNS'}")
            
            # Check for critical issues
            critical_issues = []
            if dq['data_structure']['duplicate_records'] > 0:
                critical_issues.append(f"Duplicate records: {dq['data_structure']['duplicate_records']}")
            
            high_missing = [col for col, data in dq.get('missing_data', {}).items() if data['severity'] == 'high']
            if high_missing:
                critical_issues.append(f"High missing data: {high_missing}")
            
            invalid_ranges = [var for var, data in dq.get('range_validation', {}).items() if not data['is_valid']]
            if invalid_ranges:
                critical_issues.append(f"Invalid ranges: {invalid_ranges}")
            
            if critical_issues:
                print(f"  Critical Issues: {', '.join(critical_issues)}")
            else:
                print(f"  No critical data quality issues detected")
        
        # Statistical assumptions summary
        if 'statistical_assumptions' in self.validation_results:
            sa = self.validation_results['statistical_assumptions']
            assumptions_met = 0
            total_assumptions = 0
            
            for assumption, data in sa.items():
                if 'is_' in data:
                    total_assumptions += 1
                    if data['is_' + assumption.split('_')[0]]:  # Extract assumption name
                        assumptions_met += 1
            
            print(f"  Statistical Assumptions: {assumptions_met}/{total_assumptions} met")
        
        # Recommendations
        print(f"\nRECOMMENDATIONS:")
        
        if 'data_quality' in self.validation_results:
            dq = self.validation_results['data_quality']
            
            if dq['data_structure']['duplicate_records'] > 0:
                print(f"  - Remove duplicate records before analysis")
            
            high_missing = [col for col, data in dq.get('missing_data', {}).items() if data['severity'] == 'high']
            if high_missing:
                print(f"  - Consider imputation or exclusion for variables with high missing data: {high_missing}")
            
            high_outliers = [col for col, data in dq.get('outliers', {}).items() if data['percentage'] > 10]
            if high_outliers:
                print(f"  - Investigate outliers in: {high_outliers}")
        
        if 'statistical_assumptions' in self.validation_results:
            sa = self.validation_results['statistical_assumptions']
            
            if not sa.get('normality', {}).get('is_normal', True):
                print(f"  - Consider non-parametric alternatives or data transformation")
            
            if not sa.get('homoscedasticity', {}).get('is_homoscedastic', True):
                print(f"  - Consider robust standard errors or weighted regression")
            
            if sa.get('multicollinearity', {}).get('max_vif', 0) > 10:
                print(f"  - Address multicollinearity by removing or combining variables")
        
        print(f"\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        
        return self.validation_results
    
    def run_comprehensive_validation(self):
        """
        Run all validation checks
        """
        print("Starting comprehensive WAI analysis validation...")
        
        # Data quality validation
        self.validate_data_quality()
        
        # Statistical assumptions validation
        self.validate_statistical_assumptions()
        
        # Generate final report
        return self.generate_validation_report()


def validate_wai_analysis(df, analysis_results=None):
    """
    Convenience function to validate WAI analysis
    
    Args:
        df (pd.DataFrame): The dataset used for analysis
        analysis_results (dict): Optional analysis results to validate
    
    Returns:
        dict: Comprehensive validation results
    """
    validator = WAIAnalysisValidator(df, analysis_results)
    return validator.run_comprehensive_validation()


if __name__ == "__main__":
    # Example usage
    print("WAI Analysis Validation Module")
    print("Use this module to validate your WAI analysis results")
    print("Example: validator = WAIAnalysisValidator(your_dataframe)")
    print("         results = validator.run_comprehensive_validation()") 