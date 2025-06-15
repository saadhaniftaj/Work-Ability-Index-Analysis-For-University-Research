# Work Ability Index (WAI) Analysis Tool

A comprehensive web-based application for analyzing Work Ability Index data from Excel files with advanced statistical analysis capabilities. This tool was developed for university research purposes to examine the relationship between work ability and various psychological, physical, and demographic factors.

## 🎯 Project Overview

This application provides a sophisticated analysis platform for researchers studying work ability, return-to-work outcomes, and their correlations with psychological assessments, physical measurements, and demographic variables. The tool processes Excel files containing participant data and generates comprehensive statistical reports.

## 🔬 Research Methodology

### Core Analysis Requirements

The application performs six key statistical analyses as required for university research:

1. **Multiple Regression Analysis**
   - Examines relationships between WAI scores and psychological factors
   - Variables: GAD-7 (anxiety), Social Support Scale, Brief Resilience Scale (BRS)
   - Provides R², adjusted R², F-statistics, and coefficient significance

2. **T-Test Analysis**
   - Compares high vs low WAI groups across psychological instruments
   - Identifies significant differences in anxiety, social support, and resilience
   - Reports effect sizes and confidence intervals

3. **WAI < 27 Analysis with Return-to-Work**
   - Identifies participants with low work ability (WAI < 27)
   - Analyzes return-to-work status for this vulnerable group
   - Provides demographic breakdown and intervention recommendations

4. **Demographic Factors Analysis**
   - Examines effects of age, gender, weight, and injury factors on WAI
   - Controls for confounding variables
   - Identifies significant demographic predictors

5. **Pain & Blood Pressure Effects**
   - Analyzes correlations between pain levels, blood pressure, and work ability
   - Examines pain as a mediator variable
   - Provides clinical implications for pain management

6. **Grip Strength Correlation**
   - Examines relationship between grip strength and work ability
   - Analyzes both right and left grip strength
   - Provides physical assessment insights

## 🎨 User Interface Features

### Modern Design
- **Dark Theme**: Professional gradient-based interface optimized for research environments
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Interactive Elements**: Hover effects, animations, and smooth transitions
- **Accessibility**: High contrast design with clear typography

### File Management
- **Drag-and-Drop Upload**: Intuitive file selection interface
- **Multiple File Support**: Process up to 20 Excel files simultaneously
- **Real-time Validation**: Immediate feedback on file format and size
- **Progress Tracking**: Visual indicators during analysis

### Results Display
- **Comprehensive Reporting**: Detailed statistical analysis in downloadable text format
- **Error Handling**: Robust error detection with user-friendly messages
- **Download Functionality**: Easy access to generated reports
- **Session Management**: Maintains analysis state across interactions

## 📊 Data Requirements

### File Format Specifications
- **Supported Formats**: Excel (.xlsx, .xls)
- **Header Row**: Row 3 (index 2) - critical for proper data mapping
- **Data Start**: Row 4 (index 3) - actual participant data begins here
- **Maximum Files**: 20 per analysis session
- **File Size Limit**: 100MB per file

### Required Column Mapping
The application automatically maps these column names to standardized variables:

| Original Column | Standardized Variable | Description |
|----------------|---------------------|-------------|
| `WAI score` | Work Ability Index | Primary outcome measure |
| `GAD-7` | Anxiety Assessment | Generalized Anxiety Disorder scale |
| `Social Support` | Social Support Scale | Perceived social support measure |
| `BRS` | Brief Resilience Scale | Psychological resilience measure |
| `Grip R` / `Grip L` | Right/Left Grip Strength | Physical strength assessment |
| `Pain Level 1-10` | Pain Assessment | Subjective pain rating |
| `BP Final` | Blood Pressure | Cardiovascular measure |
| `Weight` | Body Weight | Anthropometric measure |
| `Sex` | Gender | Demographic variable |
| `DOB` | Date of Birth | Age calculation |
| `Date of Injury` | Injury Date | Time since injury calculation |
| `BODY PART` | Type of Injury | Injury classification |
| `RTW in FCE` | Return to Work Status | Functional capacity evaluation outcome |

## 🚀 Deployment Options

### 1. Local Development Environment
```bash
# Clone the repository
git clone https://github.com/saadhaniftaj/Work-Ability-Index-Analysis-For-University-Research.git
cd Work-Ability-Index-Analysis-For-University-Research

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Access at http://127.0.0.1:5000
```

### 2. AWS App Runner Deployment
- Uses `Dockerfile.apprunner` for containerized deployment
- Configure via `apprunner.yaml` for automated scaling
- Deploy using AWS CLI or console interface
- Automatic HTTPS and load balancing

### 3. Heroku Cloud Platform
- Uses `Procfile` for process management
- `runtime.txt` specifies Python 3.9+ compatibility
- Automatic deployment from Git repository
- Built-in logging and monitoring

### 4. Docker Containerization
```bash
# Build Docker image
docker build -t wai-analysis .

# Run container
docker run -p 5000:5000 wai-analysis

# Using Docker Compose
docker-compose up -d
```

## 📁 Project Structure

```
Work-Ability-Index-Analysis-For-University-Research/
├── app.py                          # Main Flask application
├── enhanced_wai_analysis_final.py  # Core analysis engine
├── wai_validation.py               # Data validation utilities
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
├── Dockerfile.apprunner           # AWS App Runner configuration
├── Procfile                       # Heroku process file
├── runtime.txt                    # Python runtime specification
├── README.md                      # This documentation
├── templates/                     # HTML templates
│   ├── index.html                 # Upload interface
│   └── results.html               # Results display
├── uploads/                       # File upload directory
├── reports/                       # Generated reports directory
└── logs/                          # Application logs
```

## 🔧 Technical Dependencies

### Core Libraries
- **Flask 2.0+**: Web framework for application server
- **pandas**: Advanced data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scipy**: Statistical analysis and hypothesis testing
- **scikit-learn**: Machine learning algorithms
- **openpyxl**: Excel file reading and writing
- **matplotlib**: Data visualization and plotting
- **seaborn**: Statistical data visualization

### Statistical Analysis Capabilities
- Multiple regression analysis with diagnostics
- T-tests and ANOVA for group comparisons
- Correlation analysis (Pearson, Spearman)
- Descriptive statistics and data summaries
- Effect size calculations
- Confidence interval estimation

## 📈 Performance Features

### Optimization
- **Efficient Memory Usage**: Optimized for large datasets
- **Fast Processing**: Vectorized operations for statistical calculations
- **Concurrent Handling**: Multiple file processing capabilities
- **Caching**: Intelligent result caching for repeated analyses

### Scalability
- **Horizontal Scaling**: Containerized deployment supports multiple instances
- **Load Balancing**: AWS App Runner provides automatic scaling
- **Resource Management**: Efficient CPU and memory utilization

## 🔒 Security Features

### Data Protection
- **File Type Validation**: Strict Excel file format checking
- **Size Limits**: Prevents oversized file uploads
- **Secure Handling**: Temporary file storage with automatic cleanup
- **Input Sanitization**: Protection against malicious inputs
- **Error Filtering**: Prevents sensitive information leakage

### Privacy Compliance
- **Local Processing**: Data processed on server, not transmitted externally
- **Temporary Storage**: Files automatically deleted after analysis
- **No Data Persistence**: Results generated on-demand, not stored permanently

## 📋 Usage Instructions

### Step-by-Step Process

1. **Access Application**: Navigate to the deployed URL or localhost:5000
2. **Upload Files**: Select Excel files using the web interface
3. **Validation**: System automatically validates file format and structure
4. **Analysis**: Click "Analyze Files" to start statistical processing
5. **Review Results**: View comprehensive analysis summary
6. **Download Report**: Generate and download detailed text report
7. **Repeat**: Upload additional files for comparative analysis

### Best Practices

- **Data Preparation**: Ensure Excel files have headers in row 3
- **Column Naming**: Use exact column names as specified in requirements
- **Data Quality**: Check for missing values and outliers before upload
- **Sample Size**: Ensure adequate sample sizes for statistical power
- **Backup**: Keep original data files as backup

## 🐛 Error Handling

The application includes comprehensive error handling for:

- **File Format Issues**: Invalid Excel files or corrupted data
- **Missing Columns**: Required variables not found in dataset
- **Data Quality Problems**: Insufficient sample sizes or invalid values
- **Processing Errors**: Statistical analysis failures
- **Network Issues**: Connectivity problems during upload/download

## 📞 Support and Documentation

### Technical Support
- **Logging**: Comprehensive application logs for debugging
- **Error Messages**: User-friendly error descriptions
- **Validation Feedback**: Real-time file validation results

### Research Methodology
- **Statistical Documentation**: Detailed explanation of analysis methods
- **Interpretation Guidelines**: Help with understanding results
- **Clinical Implications**: Practical applications of findings

## 🔄 Version History

- **v10.0** (Current): Enhanced deployment package with AWS App Runner support
- **v9.0**: Text-only report generation for research compliance
- **v8.0**: Accurate statistical analysis with proper data handling
- **v7.0**: Web interface with modern design
- **v6.0**: Core analysis engine development

## 📄 License and Attribution

This project was developed for university research purposes. Please ensure proper attribution when using this tool in academic publications or research reports.

---

**Developed for**: University Research Project  
**Version**: v10.0  
**Last Updated**: June 2025  
**Compatibility**: Python 3.8+, Flask 2.0+  
**Deployment**: AWS App Runner, Heroku, Docker, Local Development