# IoT Side-Channel Password Inference System

[![Platform](https://img.shields.io/badge/Platform-Android-green.svg)](https://developer.android.com/)
[![Language](https://img.shields.io/badge/Language-Kotlin-purple.svg)](https://kotlinlang.org/)
[![ML](https://img.shields.io/badge/ML-XGBoost-blue.svg)](https://xgboost.readthedocs.io/)
[![Accuracy](https://img.shields.io/badge/Accuracy-100%25-brightgreen.svg)](https://github.com/East-Hu/IOT-FALL-2025)

A comprehensive research project demonstrating privacy vulnerabilities in mobile devices through side-channel attacks. This system uses motion sensor data to predict password inputs with state-of-the-art machine learning techniques.

## Project Overview

This project addresses the critical security concern of side-channel attacks on mobile devices. By leveraging embedded sensors (accelerometer, gyroscope, rotation vector, and magnetometer), we demonstrate how seemingly innocuous sensor data can be exploited to infer sensitive user input such as numeric passwords.

### Key Achievements

- **Training Accuracy**: 98.75%
- **Testing Accuracy**: 100% (10/10 correct predictions)
- **Feature Dimensions**: 303-dimensional feature space
- **Model Performance**: Successfully handles all edge cases including repeated digits
- **Data Quality**: 1,605 high-quality synthetic training samples

## Table of Contents

- [System Architecture](#system-architecture)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Android App](#android-app)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Results](#results)
- [Security Implications](#security-implications)
- [Documentation](#documentation)
- [License](#license)

## System Architecture

The system consists of two main components:

### 1. Android Application (Data Collection)

- **Purpose**: Collects sensor data during password input
- **Technology**: Kotlin, Jetpack Compose, Android Services
- **Sensors**: Accelerometer, Gyroscope, Rotation Vector, Magnetometer
- **Sampling Rate**: 50 Hz (20,000 microseconds)
- **Features**:
  - Background data collection
  - Real-time label assignment
  - CSV file export
  - Foreground service for continuous operation

### 2. Machine Learning Pipeline (Password Inference)

- **Purpose**: Trains models and predicts passwords from sensor data
- **Technology**: Python, XGBoost, scikit-learn, pandas
- **Pipeline Stages**:
  1. Data preprocessing with time-interval segmentation
  2. Feature extraction (time-domain + frequency-domain)
  3. Model training with hyperparameter optimization
  4. Password prediction with confidence scoring

## Features

### Android Application

- **Dual Mode Operation**:
  - Sensor Data Collector: General-purpose sensor data recording
  - Password Prediction Mode: Specialized training data collection with numeric keypad

- **Robust Data Collection**:
  - Start/Stop controls
  - Filename customization
  - Background operation support
  - Real-time data point counter

- **Sensor Integration**:
  - Accelerometer (device motion)
  - Gyroscope (rotation rate)
  - Rotation Vector (device orientation)
  - Magnetometer (magnetic field)

### Machine Learning System

- **Advanced Preprocessing**:
  - Label-based segmentation
  - Time-interval detection for repeated digits (500ms threshold)
  - Automated data cleaning and validation

- **Comprehensive Feature Extraction** (303 features):
  - **Time-domain**: mean, std, max, min, median, range, RMS, skewness, kurtosis, percentiles, zero-crossing rate
  - **Frequency-domain**: FFT energy, spectral mean/std, dominant frequency, spectral centroid

- **Multiple Model Support**:
  - XGBoost (Recommended - Best accuracy)
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression

- **Automated Testing**:
  - Batch prediction evaluation
  - Confusion matrix generation
  - Feature importance analysis
  - Comprehensive reporting

## Requirements

### Android Application

- Android Studio Arctic Fox or later
- Android SDK 34 (API Level 34)
- Minimum Android SDK 24 (API Level 24)
- Kotlin 1.9+
- Physical Android device with motion sensors

### Machine Learning Pipeline

- Python 3.8+
- Required packages:
  ```
  pandas>=1.3.0
  numpy>=1.21.0
  scikit-learn>=1.0.0
  xgboost>=1.5.0
  scipy>=1.7.0
  matplotlib>=3.4.0
  seaborn>=0.11.0
  joblib>=1.1.0
  ```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/East-Hu/IOT-FALL-2025.git
cd IOT-FALL-2025
```

### 2. Android App Setup

```bash
# Open the project in Android Studio
# Wait for Gradle sync to complete
# Connect your Android device via USB with USB debugging enabled
# Build and run the application
```

### 3. Python Environment Setup

```bash
cd ml_code

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost scipy matplotlib seaborn joblib
```

## Usage

### Android App

#### Mode 1: General Data Collection

1. Launch the application
2. Enter a filename (must end with `.csv`)
3. Tap "开始收集" (Start Collection) to begin
4. Perform desired activities
5. Tap "停止收集" (Stop Collection) to save data

#### Mode 2: Password Training Data Collection

1. Tap "进入密码预测模式" (Enter Password Prediction Mode)
2. Use the numeric keypad (0-9) to input passwords
3. Data collection starts automatically on first key press
4. Continue entering password digits (each digit is automatically labeled)
5. Tap "完成并保存" (Complete and Save) when finished

#### Exporting Data

```bash
# Connect device via ADB
adb devices

# Export collected data
adb pull /sdcard/Android/data/com.example.iotproject/files/ ./sensor_data/

# Or use the provided script
./export_data.sh
```

### Machine Learning Pipeline

#### Quick Start: Complete Pipeline

```bash
cd ml_code

# Run the entire pipeline with one command
python run_all.py --data_dir ../sensor_data/files --model xgboost
```

#### Step-by-Step Execution

##### Step 1: Data Preprocessing

```bash
python 1_data_preprocessing.py \
    --data_dir ../sensor_data/files \
    --output ./processed_data
```

Segments raw CSV files into individual key press events with time-interval detection.

##### Step 2: Feature Extraction

```bash
python 2_feature_extraction.py \
    --input ./processed_data \
    --output ./features.csv
```

Extracts 303-dimensional feature vectors from sensor data.

##### Step 3: Model Training

```bash
# XGBoost (recommended for best accuracy)
python 3_train_model.py \
    --features ./features.csv \
    --model xgboost

# Random Forest (good balance of speed and accuracy)
python 3_train_model.py \
    --features ./features.csv \
    --model random_forest
```

Trains the classification model and generates performance visualizations.

##### Step 4: Password Prediction

```bash
python 4_predict_password.py \
    --model ./models/xgboost_YYYYMMDD_HHMMSS.pkl \
    --data ../test_data/test_password_12345_*.csv \
    --actual 12345
```

Predicts password from new sensor data and calculates accuracy.

#### Automated Testing

```bash
# Run automated test suite on all test passwords
python run_auto_test.py
```

This evaluates the model on 10 pre-defined test passwords and generates a comprehensive report.

## Project Structure

```
iotproject/
│
├── app/                                    # Android application
│   └── src/main/java/com/example/iotproject/
│       ├── MainActivity.kt                 # Main UI (data collector + password input)
│       ├── SensorService.kt                # Background sensor data collection
│       └── ui/theme/                       # UI theming
│
├── ml_code/                                # Machine learning pipeline
│   ├── 1_data_preprocessing.py            # CSV segmentation by key press events
│   ├── 2_feature_extraction.py            # Feature engineering (time + frequency domain)
│   ├── 3_train_model.py                   # Model training with multiple algorithms
│   ├── 4_predict_password.py              # Password prediction and evaluation
│   ├── run_all.py                          # Automated pipeline execution
│   ├── run_auto_test.py                    # Automated testing suite
│   ├── generate_synthetic_data_v2.py      # High-quality training data generation
│   ├── generate_test_data.py              # Test data generation
│   ├── test_fix.py                         # Repeated digit fix verification
│   ├── models/                             # Trained model files (.pkl)
│   ├── processed_data/                     # Preprocessed key press segments
│   └── README.md                           # ML pipeline documentation
│
├── sensor_data/                            # Raw sensor data (collected from app)
│   └── files/                              # CSV files with sensor readings
│
├── sensor_data_synthetic/                  # High-quality synthetic training data
│   └── files/                              # 1,605 training samples
│
├── test_data/                              # Test password data (15 passwords)
│   ├── test_password_12345_*.csv
│   ├── test_password_11111_*.csv
│   └── ...
│
├── test_results/                           # Test reports and performance metrics
│
├── md/                                     # Chinese documentation
│   ├── 项目最终完成总结.md
│   ├── 测试结果展示报告.md
│   ├── 重复数字识别修复说明.md
│   └── ...
│
├── build.gradle.kts                        # Android project configuration
├── settings.gradle.kts
├── export_data.sh                          # ADB data export script
├── clean_data.sh                           # Data cleaning utilities
└── README.md                               # This file
```

## Technical Details

### Sensor Data Format

Each CSV file contains the following columns:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | long | System time in milliseconds |
| nanoTime | long | High-resolution timestamp in nanoseconds |
| type | string | Sensor type (ACC, GYRO, ROT_VEC, MAG) |
| x | float | X-axis reading |
| y | float | Y-axis reading |
| z | float | Z-axis reading |
| w | float | W component (rotation vector only) |
| label | string | Current digit being pressed (0-9) |

### Feature Engineering

The system extracts comprehensive features from each sensor reading:

**Time-Domain Features (per axis, per sensor)**:
- Statistical: mean, standard deviation, min, max, median, range
- Distribution: skewness, kurtosis, percentiles (25%, 50%, 75%)
- Signal: RMS (root mean square), zero-crossing rate

**Frequency-Domain Features (per axis, per sensor)**:
- FFT energy distribution
- Spectral statistics: mean, standard deviation
- Dominant frequency
- Spectral centroid

**Total**: 4 sensors × (3-4 axes) × (11 time + 7 frequency features) ≈ **303 features**

### Machine Learning Models

#### XGBoost (Primary Model)

- **Type**: Gradient Boosting Decision Trees
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 6
  - learning_rate: 0.1
  - subsample: 0.8
- **Advantages**: Highest accuracy, feature importance, handles non-linear relationships

#### Random Forest (Alternative)

- **Type**: Ensemble of Decision Trees
- **Hyperparameters**:
  - n_estimators: 100
  - max_depth: 20
  - min_samples_split: 2
- **Advantages**: Fast training, interpretable, resistant to overfitting

### Key Technical Innovations

1. **Time-Interval Segmentation**: Solves the repeated digit problem by detecting key press boundaries using 500ms time gaps, in addition to label changes.

2. **Synthetic Data Generation**: Creates high-quality training data with realistic sensor patterns and temporal characteristics.

3. **Multi-Sensor Fusion**: Combines data from four different sensors to capture comprehensive device motion patterns.

## Results

### Model Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 98.75% |
| Cross-Validation Score | 99.84% |
| Test Accuracy | 100% (10/10) |
| F1 Score (weighted) | 0.9875 |

### Test Password Results

All 10 test passwords were correctly predicted:

| # | Password | Type | Accuracy | Status |
|---|----------|------|----------|--------|
| 1 | 12345 | Sequential | 100% | ✓ |
| 2 | 54321 | Reverse | 100% | ✓ |
| 3 | 13579 | Odd numbers | 100% | ✓ |
| 4 | 24680 | Even numbers | 100% | ✓ |
| 5 | 11111 | Repeated | 100% | ✓ |
| 6 | 98765 | High digits | 100% | ✓ |
| 7 | 02468 | Starting with 0 | 100% | ✓ |
| 8 | 19283 | Random | 100% | ✓ |
| 9 | 74650 | Random | 100% | ✓ |
| 10 | 36912 | Random | 100% | ✓ |

### Performance Visualizations

The training process generates:
- Confusion matrix showing per-digit accuracy
- Feature importance ranking
- Training/validation curves
- Detailed test reports

## Security Implications

### Demonstrated Vulnerabilities

This research demonstrates that:

1. **Motion sensors pose significant privacy risks** - Sensor data can reveal sensitive user input without explicit permissions.

2. **Background sensor access is exploitable** - Malicious apps can collect sensor data while running in the background.

3. **High prediction accuracy is achievable** - With sufficient training data, side-channel attacks can be highly effective.

### Defense Recommendations

1. **User Awareness**: Users should be cautious about granting sensor permissions to untrusted applications.

2. **Operating System Protections**:
   - Restrict background sensor access for non-foreground apps
   - Implement sensor permission controls similar to camera/microphone
   - Add random delays or noise to sensor readings for sensitive operations

3. **Application Design**:
   - Use biometric authentication instead of numeric PINs
   - Implement randomized keyboard layouts
   - Add artificial delays between key presses
   - Use haptic feedback to mask motion patterns

4. **Hardware Solutions**:
   - Sensor usage indicators (similar to camera LED)
   - Hardware-level sensor access controls

## Documentation

Comprehensive documentation is available in the `md/` directory (in Chinese):

- `项目最终完成总结.md` - Complete project summary with achievements
- `重复数字识别修复说明.md` - Technical details on repeated digit recognition fix
- `测试结果展示报告.md` - Detailed test results and analysis
- `测试使用说明.md` - Testing guide and instructions
- `数据质量分析报告.md` - Data quality analysis report

Machine learning documentation:
- `ml_code/README.md` - ML pipeline usage guide
- `ml_code/特征说明.md` - Feature engineering documentation
- `ml_code/训练过程说明.md` - Training process explanation

## Contributing

This is an academic research project completed as part of CSC 8223 - Internet of Things coursework. The code is provided for educational and research purposes.

## Acknowledgments

- Course: CSC 8223 - Internet of Things
- Institution: [Your Institution Name]
- Project Type: Privacy Leakage Research on Mobile Devices

## License

This project is provided for academic and research purposes. Please cite this work if you use it in your research.

## Contact

For questions or collaboration inquiries, please open an issue on the [GitHub repository](https://github.com/East-Hu/IOT-FALL-2025).

---

**Disclaimer**: This project is intended solely for educational and research purposes to raise awareness about mobile device security vulnerabilities. Unauthorized use of these techniques against real users without consent is illegal and unethical.
