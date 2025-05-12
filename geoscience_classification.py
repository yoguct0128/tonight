import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os

def load_excel_data(file_path, sheet_name):
    """Load Excel data from specified file and sheet"""
    return pd.read_excel(file_path, sheet_name=sheet_name)

def train_models(X_train, y_train):
    """Train and optimize SVM and Random Forest models using grid search"""
    # SVM hyperparameter grid search
    svm_params = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    svm = SVC(probability=True)
    svm_grid = GridSearchCV(svm, svm_params, cv=5)
    svm_grid.fit(X_train, y_train)
    
    # Random Forest hyperparameter grid search
    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestClassifier()
    rf_grid = GridSearchCV(rf, rf_params, cv=5)
    rf_grid.fit(X_train, y_train)
    
    return svm_grid.best_estimator_, rf_grid.best_estimator_

def balance_dataset(X, y):
    """Handle class imbalance using SMOTE oversampling"""
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def plot_roc_curve(ax, model, X_train, y_train, X_test, y_test, label, color, linestyle, is_first):
    """Plot ROC curves for both training and test datasets"""
    # Training set ROC
    y_pred_proba_train = model.predict_proba(X_train)[:, 1]
    fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_proba_train)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    # Test set ROC
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]
    fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_proba_test)
    roc_auc_test = auc(fpr_test, tpr_test)
    
    print(f'{label} - Train AUC: {roc_auc_train:.3f}, Test AUC: {roc_auc_test:.3f}')
    
    # Plot ROC curves
    ax.plot(fpr_train, tpr_train, label=f'{label} Train (area = {roc_auc_train:.2f})', color=color, linestyle=linestyle)
    ax.plot(fpr_test, tpr_test, label=f'{label} Test (area = {roc_auc_test:.2f})', color=color, linestyle='--')
    
    if is_first:
        ax.plot([0, 1], [0, 1], 'k--', label='Chance level')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'Receiver Operating Characteristic (ROC) Curve - {label}')
        ax.legend(loc="lower right")

def evaluate_models(X_balanced_scaled, y_balanced):
    """Evaluate model performance with different test set ratios"""
    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    # Create ROC curve plots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    best_svm_models = []
    best_rf_models = []
    
    for i, test_size in enumerate(test_sizes):
        # Split dataset into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_balanced_scaled, y_balanced, test_size=test_size, random_state=42
        )
        
        # Train models
        svm_model, rf_model = train_models(X_train, y_train)
        
        # Save best models
        best_svm_models.append(svm_model)
        best_rf_models.append(rf_model)
        
        # Test set performance evaluation
        print(f"SVM Test Classification Report (Test Size: {test_size})")
        print(classification_report(y_test, svm_model.predict(X_test)))
        print(f"Random Forest Test Classification Report (Test Size: {test_size})")
        print(classification_report(y_test, rf_model.predict(X_test)))
        
        # Plot ROC curves
        plot_roc_curve(axs[0], svm_model, X_train, y_train, X_test, y_test, 
                      'SVM', colors[i], '-', i == 0)
        plot_roc_curve(axs[1], rf_model, X_train, y_train, X_test, y_test, 
                      'Random Forest', colors[i], '-', i == 0)
    
    plt.tight_layout()
    return best_svm_models, best_rf_models, test_sizes

def predict_samples(models, scaler, prediction_data, test_sizes, model_type):
    """Predict samples using trained models"""
    X_pred = prediction_data.iloc[:, 1:14].values
    X_pred_scaled = scaler.transform(X_pred)
    
    # Predict using models trained with different test set ratios
    for i, test_size in enumerate(test_sizes):
        model = models[i]
        predictions = model.predict_proba(X_pred_scaled)[:, 1]
        prediction_data[f'{model_type}_Probability_{test_size}'] = predictions
    
    return prediction_data

def process_data_type(data_type, base_path):
    """Process data for a specific data type (CF, FR, IV)"""
    print(f"\nProcessing data type: {data_type}")
    
    # Load dataset
    negative_samples = load_excel_data(os.path.join(base_path, 'Negative.xls'), data_type)
    positive_samples = load_excel_data(os.path.join(base_path, 'Positive.xls'), data_type)
    
    # Prepare features and labels
    X_positive = positive_samples.iloc[:, 3:15].values
    y_positive = [1] * len(X_positive)
    X_negative = negative_samples.iloc[:, 3:15].values
    y_negative = [0] * len(X_negative)
    
    # Combine positive and negative samples
    X = np.concatenate((X_positive, X_negative), axis=0)
    y = np.concatenate((y_positive, y_negative), axis=0)
    
    # Handle class imbalance
    print("Handling class imbalance...")
    X_balanced, y_balanced = balance_dataset(X, y)
    
    # Feature standardization
    print("Standardizing features...")
    scaler = StandardScaler()
    X_balanced_scaled = scaler.fit_transform(X_balanced)
    
    # Model evaluation
    print("Training and evaluating models...")
    best_svm_models, best_rf_models, test_sizes = evaluate_models(X_balanced_scaled, y_balanced)
    
    # Create ROC curve plot directory if not exists
    roc_dir = os.path.join(base_path, 'roc_plots')
    os.makedirs(roc_dir, exist_ok=True)
    
    # Save ROC curve plot
    plt.savefig(os.path.join(roc_dir, f'roc_curve_{data_type}.png'))
    plt.close()
    
    # Predict new samples
    print("Predicting on new samples...")
    prediction_data = load_excel_data(os.path.join(base_path, 'TotalSamples.xlsx'), data_type)
    prediction_data = predict_samples(best_svm_models, scaler, prediction_data, test_sizes, 'SVM')
    prediction_data = predict_samples(best_rf_models, scaler, prediction_data, test_sizes, 'RF')
    
    # Save prediction results
    output_path = os.path.join(base_path, f'prediction_results_{data_type}.xlsx')
    prediction_data.to_excel(output_path, index=False)
    print(f"Prediction results saved to: {output_path}")

def main():
    """Main program entry point"""
    try:
        base_path = r'C:\Users\Administrator\Desktop\Pingjiang_manuscript'
        data_types = ['CF', 'FR', 'IV']
        
        for data_type in data_types:
            process_data_type(data_type, base_path)
        
        print("\nAll data types processed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()    