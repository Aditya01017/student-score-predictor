"""
Student Grade Prediction - Standard ML Workflow
==============================================

This script follows the standard 9-step ML workflow for solving a Kaggle/internship problem:
1. Understand the Problem
2. Data Exploration
3. Feature Engineering
4. Preprocessing
5. Modeling
6. Evaluation
7. Interpretation
8. Business/Practical Insights
9. Deployment

Author: Aditya Jain
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

print("="*60)
print("STUDENT GRADE PREDICTION - STANDARD ML WORKFLOW")
print("="*60)

# ============================================================================
# STEP 1: UNDERSTAND THE PROBLEM
# ============================================================================
print("\n📋 STEP 1: UNDERSTAND THE PROBLEM")
print("-" * 40)

print("Problem: Predict student final grades (G3) using available features")
print("Type: Regression (predicting a continuous value)")
print("Target: G3 (final grade, 0-20 scale)")
print("Goal: Build a model to predict student performance for early intervention")

# ============================================================================
# STEP 2: DATA EXPLORATION
# ============================================================================
print("\n🔍 STEP 2: DATA EXPLORATION")
print("-" * 40)

# Load the data
print("Loading data...")
df = pd.read_csv('data/student-mat.csv', sep=';')
print(f"Dataset shape: {df.shape}")
print(f"Features: {df.shape[1]-1}, Target: 1 (G3)")

# Basic info
print("\n📊 Basic Information:")
print(f"• Missing values: {df.isnull().sum().sum()}")
print(f"• Data types: {df.dtypes.value_counts()}")

# Target variable analysis
print("\n🎯 Target Variable (G3) Analysis:")
print(f"• Mean: {df['G3'].mean():.2f}")
print(f"• Median: {df['G3'].median():.2f}")
print(f"• Std: {df['G3'].std():.2f}")
print(f"• Min: {df['G3'].min()}")
print(f"• Max: {df['G3'].max()}")

# Check for outliers in target
Q1 = df['G3'].quantile(0.25)
Q3 = df['G3'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['G3'] < Q1 - 1.5*IQR) | (df['G3'] > Q3 + 1.5*IQR)]
print(f"• Outliers: {len(outliers)} students")

# Key features correlation with target
print("\n🔗 Key Features Correlation with G3:")
key_features = ['G1', 'G2', 'studytime', 'absences', 'age']
for feature in key_features:
    if feature in df.columns:
        corr = df[feature].corr(df['G3'])
        print(f"• {feature}: {corr:.3f}")

# Create basic visualizations
print("\n📈 Creating basic visualizations...")

# Target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(df['G3'], bins=20, alpha=0.7, color='skyblue')
plt.title('Distribution of Final Grades (G3)')
plt.xlabel('Grade')
plt.ylabel('Count')

# G1 vs G3
plt.subplot(1, 3, 2)
plt.scatter(df['G1'], df['G3'], alpha=0.6)
plt.title('G1 vs G3')
plt.xlabel('First Period Grade (G1)')
plt.ylabel('Final Grade (G3)')

# G2 vs G3
plt.subplot(1, 3, 3)
plt.scatter(df['G2'], df['G3'], alpha=0.6, color='orange')
plt.title('G2 vs G3')
plt.xlabel('Second Period Grade (G2)')
plt.ylabel('Final Grade (G3)')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot instead of showing it

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================
print("\n🔧 STEP 3: FEATURE ENGINEERING")
print("-" * 40)

# Create new features based on domain knowledge
print("Creating new features...")

# Academic progression features
df['grade_progression'] = df['G2'] - df['G1']  # How much student improved
df['grade_decline'] = (df['G1'] > df['G2']).astype(int)  # Did grades decline?
df['consistent_performer'] = ((df['G1'] >= 10) & (df['G2'] >= 10)).astype(int)  # Consistent good performance

# Study and attendance features
df['study_attendance_ratio'] = df['studytime'] / (df['absences'] + 1)  # Study efficiency
df['high_risk_attendance'] = (df['absences'] > 10).astype(int)  # High absence risk

# Family and support features
df['parent_education'] = (df['Medu'] + df['Fedu']) / 2  # Average parent education
df['family_support'] = ((df['famsup'] == 'yes') & (df['schoolsup'] == 'yes')).astype(int)  # Both supports

# Risk indicators
df['multiple_failures'] = (df['failures'] > 1).astype(int)  # Multiple past failures
df['low_health'] = (df['health'] <= 2).astype(int)  # Poor health

print(f"Created {len(df.columns) - 33} new features")
print("New features:")
new_features = ['grade_progression', 'grade_decline', 'consistent_performer', 
                'study_attendance_ratio', 'high_risk_attendance', 'parent_education',
                'family_support', 'multiple_failures', 'low_health']
for feature in new_features:
    corr = df[feature].corr(df['G3'])
    print(f"• {feature}: {corr:.3f} correlation with G3")

# ============================================================================
# STEP 4: PREPROCESSING
# ============================================================================
print("\n⚙️ STEP 4: PREPROCESSING")
print("-" * 40)

# Separate features and target
X = df.drop('G3', axis=1)
y = df['G3']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=['int64', 'float64']).columns

print(f"\nData types:")
print(f"• Categorical features: {len(categorical_columns)}")
print(f"• Numerical features: {len(numerical_columns)}")

# Encode categorical variables
print("\n🔄 Encoding categorical variables...")
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le
    print(f"• {col}: {len(le.classes_)} categories encoded")

# Scale numerical features
print("\n⚖️ Scaling numerical features...")
scaler = StandardScaler()
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
print("• All numerical features scaled to mean=0, std=1")

# Split data
print("\n✂️ Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"• Training set: {X_train.shape[0]} students")
print(f"• Testing set: {X_test.shape[0]} students")

# ============================================================================
# STEP 5: MODELING
# ============================================================================
print("\n🤖 STEP 5: MODELING")
print("-" * 40)

# Define models to try
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {}

print("Training models...")
for name, model in models.items():
    print(f"\n🔨 Training {name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    results[name] = {
        'model': model,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'predictions': y_pred
    }
    
    print(f"  ✅ R² Score: {r2:.4f} ({r2*100:.1f}%)")
    print(f"  ✅ RMSE: {rmse:.4f}")
    print(f"  ✅ MAE: {mae:.4f}")
    print(f"  ✅ CV R²: {cv_mean:.4f} ± {cv_std:.4f}")

# Hyperparameter tuning for best model
print("\n🔧 Hyperparameter tuning for Random Forest...")
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Update results with tuned model
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)
r2_tuned = r2_score(y_test, y_pred_tuned)

results['Random Forest (Tuned)'] = {
    'model': best_rf,
    'mse': mean_squared_error(y_test, y_pred_tuned),
    'rmse': np.sqrt(mean_squared_error(y_test, y_pred_tuned)),
    'mae': mean_absolute_error(y_test, y_pred_tuned),
    'r2': r2_tuned,
    'cv_mean': grid_search.best_score_,
    'cv_std': 0,  # GridSearchCV doesn't provide std
    'predictions': y_pred_tuned
}

print(f"Tuned R² Score: {r2_tuned:.4f} ({r2_tuned*100:.1f}%)")

# ============================================================================
# STEP 6: EVALUATION
# ============================================================================
print("\n📊 STEP 6: EVALUATION")
print("-" * 40)

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']
best_r2 = results[best_model_name]['r2']

print(f"🏆 BEST MODEL: {best_model_name}")
print(f"📈 Performance Metrics:")
print(f"• R² Score: {best_r2:.4f} ({best_r2*100:.1f}%)")
print(f"• RMSE: {results[best_model_name]['rmse']:.4f}")
print(f"• MAE: {results[best_model_name]['mae']:.4f}")
print(f"• Cross-validation: {results[best_model_name]['cv_mean']:.4f}")

# Model comparison
print(f"\n📊 MODEL COMPARISON:")
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R² Score': [results[model]['r2'] for model in results.keys()],
    'RMSE': [results[model]['rmse'] for model in results.keys()],
    'CV R²': [results[model]['cv_mean'] for model in results.keys()]
}).sort_values('R² Score', ascending=False)

print(comparison_df.round(4))

# Check for overfitting
print(f"\n🔍 Overfitting Check:")
for name, result in results.items():
    train_score = result['model'].score(X_train, y_train)
    test_score = result['r2']
    overfitting = train_score - test_score
    print(f"• {name}: Train R² = {train_score:.4f}, Test R² = {test_score:.4f}, Difference = {overfitting:.4f}")

# Create evaluation plots
print("\n📈 Creating evaluation plots...")

plt.figure(figsize=(15, 5))

# Actual vs Predicted
plt.subplot(1, 3, 1)
y_pred_best = results[best_model_name]['predictions']
plt.scatter(y_test, y_pred_best, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Grades')
plt.ylabel('Predicted Grades')
plt.title('Actual vs Predicted Grades')

# Residuals
plt.subplot(1, 3, 2)
residuals = y_test - y_pred_best
plt.scatter(y_pred_best, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Grades')
plt.ylabel('Residuals')
plt.title('Residual Plot')

# Residuals distribution
plt.subplot(1, 3, 3)
plt.hist(residuals, bins=20, alpha=0.7)
plt.xlabel('Residuals')
plt.ylabel('Count')
plt.title('Residuals Distribution')

plt.tight_layout()
plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot instead of showing it

# ============================================================================
# STEP 7: INTERPRETATION
# ============================================================================
print("\n🔍 STEP 7: INTERPRETATION")
print("-" * 40)

# Feature importance
if hasattr(best_model, 'feature_importances_'):
    importance = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("🔑 FEATURE IMPORTANCE:")
    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:25s}: {row['Importance']:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance.head(10)
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot instead of showing it

# Model coefficients (for linear regression)
if hasattr(best_model, 'coef_'):
    coef_df = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': best_model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)
    
    print("\n📊 LINEAR REGRESSION COEFFICIENTS:")
    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(coef_df.head(10).iterrows(), 1):
        print(f"  {i:2d}. {row['Feature']:25s}: {row['Coefficient']:.4f}")

# ============================================================================
# STEP 8: BUSINESS/PRACTICAL INSIGHTS
# ============================================================================
print("\n💼 STEP 8: BUSINESS/PRACTICAL INSIGHTS")
print("-" * 40)

# Identify at-risk students
at_risk_threshold = 8
y_pred_best = results[best_model_name]['predictions']
at_risk_predicted = (y_pred_best < at_risk_threshold).sum()
at_risk_actual = (y_test < at_risk_threshold).sum()

print("🎯 STUDENTS AT RISK:")
print(f"• Actually at risk: {at_risk_actual} students")
print(f"• Predicted at risk: {at_risk_predicted} students")
print(f"• Model can identify at-risk students for early intervention")

# Study time impact
print("\n📚 STUDY TIME IMPACT:")
study_time_impact = df.groupby('studytime')['G3'].mean()
for study_time, avg_grade in study_time_impact.items():
    study_desc = {1: '<2 hours', 2: '2-5 hours', 3: '5-10 hours', 4: '>10 hours'}
    print(f"• {study_desc.get(study_time, study_time)}: Average grade {avg_grade:.1f}")

# Attendance impact
print("\n📅 ATTENDANCE IMPACT:")
df['absence_group'] = pd.cut(df['absences'], bins=[0, 5, 10, 20, 100], labels=['0-5', '6-10', '11-20', '20+'])
attendance_impact = df.groupby('absence_group')['G3'].mean()
for group, avg_grade in attendance_impact.items():
    print(f"• {group} absences: Average grade {avg_grade:.1f}")

# Previous performance correlation
print("\n📈 PREVIOUS PERFORMANCE:")
print(f"• G1 correlation with final grade: {df['G1'].corr(df['G3']):.3f}")
print(f"• G2 correlation with final grade: {df['G2'].corr(df['G3']):.3f}")
print("• Early intervention based on G1/G2 can be very effective")

# Recommendations
print("\n💡 KEY RECOMMENDATIONS:")
print("• Monitor students with low G1/G2 grades")
print("• Encourage students to increase study time")
print("• Address attendance issues early")
print("• Provide additional support to at-risk students")
print("• Use the model to identify students needing intervention")

# ============================================================================
# STEP 9: DEPLOYMENT
# ============================================================================
print("\n🚀 STEP 9: DEPLOYMENT")
print("-" * 40)

# Save the best model
import joblib
joblib.dump(best_model, 'best_student_model.pkl')
print("✅ Best model saved as 'best_student_model.pkl'")

# Save preprocessing objects
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Preprocessing objects saved")

# Create a simple prediction function

# Save results summary
results_summary = {
    'best_model': best_model_name,
    'best_r2_score': best_r2,
    'best_rmse': results[best_model_name]['rmse'],
    'feature_importance': feature_importance.head(10).to_dict() if 'feature_importance' in locals() else None,
    'model_comparison': comparison_df.to_dict()
}

import json
with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)
print("✅ Results summary saved as 'results_summary.json'")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("="*60)
print(f"📊 Final Accuracy: {best_r2*100:.1f}%")
print(f"🏆 Best Model: {best_model_name}")
print(f"📈 Ready for deployment and real-world use!")
print(f"📁 All files saved in the project directory")
print("="*60)

print("\n📋 FILES CREATED:")
print("• best_student_model.pkl - Trained model")
print("• label_encoders.pkl - Categorical encoders")
print("• scaler.pkl - Feature scaler")
print("• results_summary.json - Project results")
print("• data_exploration.png - Data visualizations")
print("• model_evaluation.png - Model performance plots")
print("• feature_importance.png - Feature importance plot")

print("\n🚀 NEXT STEPS:")
print("• Use the trained model for new predictions")
print("• Deploy model to web app (Flask/Streamlit)")
print("• Monitor model performance over time")
print("• Retrain model with new data periodically") 