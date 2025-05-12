import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
import itertools
import seaborn as sns
from datetime import datetime, timedelta
import os

class RainfallLandslideModel:
    def __init__(self, file_path):
        """Initialize model parameters and load data"""
        self.file_path = file_path
        self.data = None
        self.n_range = range(10, 22)  # Range of n values: 10 to 21 days
        self.alpha_range = np.arange(0.5, 0.91, 0.01)  # Range of α values: 0.5 to 0.9, step 0.01
        self.optimal_n = None
        self.optimal_alpha = None
        self.best_auc = 0
        self.results = []
        
    def load_data(self, date_column='date', rainfall_column='rainfall', landslide_column='landslide'):
        """Load rainfall and landslide data"""
        try:
            self.data = pd.read_excel(self.file_path)
            # Convert date format
            self.data[date_column] = pd.to_datetime(self.data[date_column])
            self.data.set_index(date_column, inplace=True)
            self.rainfall_column = rainfall_column
            self.landslide_column = landslide_column
            print(f"Data loaded successfully, {len(self.data)} records in total")
        except Exception as e:
            print(f"Failed to load data: {e}")
            raise
    
    def calculate_effective_rainfall(self, n, alpha):
        """Calculate effective rainfall Rc
        Rc = Σ(α^i * R_i), where i ranges from 0 to n-1, and R_i is the rainfall i days ago
        """
        effective_rainfall = []
        for idx in range(len(self.data)):
            # Get rainfall data for current date and previous n days
            current_date = self.data.index[idx]
            start_date = current_date - timedelta(days=n-1)
            mask = (self.data.index >= start_date) & (self.data.index <= current_date)
            rainfall_window = self.data.loc[mask, self.rainfall_column].values
            
            # Pad with zeros if data is insufficient
            if len(rainfall_window) < n:
                rainfall_window = np.pad(rainfall_window, (n - len(rainfall_window), 0), 'constant')
            
            # Calculate effective rainfall
            weights = np.array([alpha**i for i in range(n)])
            rc = np.sum(rainfall_window * weights[::-1])  # Reverse weights to match time order
            effective_rainfall.append(rc)
        
        return np.array(effective_rainfall)
    
    def calculate_ei_d_threshold(self, effective_rainfall, duration):
        """Calculate critical rainfall threshold using EI-D model
        EI = effective rainfall, D = rainfall duration
        """
        # Simplified EI-D model, may need more complex formula in practice
        return effective_rainfall / duration
    
    def evaluate_model(self, n, alpha):
        """Evaluate model performance using AUC as the metric"""
        # Calculate effective rainfall
        effective_rainfall = self.calculate_effective_rainfall(n, alpha)
        
        # Calculate rainfall duration (simplified as n days)
        duration = np.full_like(effective_rainfall, n)
        
        # Calculate critical rainfall thresholds
        thresholds = self.calculate_ei_d_threshold(effective_rainfall, duration)
        
        # Get actual landslide occurrences
        actual_landslide = self.data[self.landslide_column].values
        
        # Calculate predicted probabilities (normalized threshold values)
        max_threshold = np.max(thresholds)
        if max_threshold > 0:
            predicted_prob = thresholds / max_threshold
        else:
            predicted_prob = np.zeros_like(thresholds)
        
        # Calculate AUC score
        try:
            auc = roc_auc_score(actual_landslide, predicted_prob)
        except ValueError:
            # Handle cases with only one class
            auc = 0.5
        
        # Calculate confusion matrix (using 0.5 as threshold)
        predicted_binary = (predicted_prob > 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(actual_landslide, predicted_binary).ravel()
        
        # Calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        return {
            'n': n,
            'alpha': alpha,
            'auc': auc,
            'accuracy': accuracy,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def grid_search_optimization(self):
        """Perform grid search to find optimal parameter combination"""
        print("Starting grid search for optimal parameters...")
        total_combinations = len(self.n_range) * len(self.alpha_range)
        progress_count = 0
        
        for n in self.n_range:
            for alpha in self.alpha_range:
                # Evaluate model performance
                result = self.evaluate_model(n, alpha)
                self.results.append(result)
                
                # Update optimal parameters
                if result['auc'] > self.best_auc:
                    self.best_auc = result['auc']
                    self.optimal_n = n
                    self.optimal_alpha = alpha
                
                # Show progress
                progress_count += 1
                if progress_count % 50 == 0:
                    progress = progress_count / total_combinations * 100
                    print(f"Progress: {progress:.2f}%, Current best parameters: n={self.optimal_n}, α={self.optimal_alpha:.2f}, AUC={self.best_auc:.4f}")
        
        print(f"Grid search completed! Optimal parameters: n={self.optimal_n}, α={self.optimal_alpha:.2f}, AUC={self.best_auc:.4f}")
        return pd.DataFrame(self.results)
    
    def plot_results_heatmap(self):
        """Plot heatmap of parameter combinations"""
        if not self.results:
            print("Please run grid search optimization first")
            return
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(self.results)
        
        # Create pivot table
        pivot_table = results_df.pivot(index='alpha', columns='n', values='auc')
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu", 
                    cbar_kws={'label': 'AUC Score'})
        plt.title('AUC Scores for Different Parameter Combinations')
        plt.xlabel('n Value (days)')
        plt.ylabel('α Value')
        plt.tight_layout()
        # Use os.path to ensure cross-platform compatibility
        output_path = os.path.join(os.getcwd(), 'parameter_heatmap.png')
        plt.savefig(output_path, dpi=300)
        plt.show()
        
    def plot_optimal_threshold(self):
        """Plot relationship between critical rainfall threshold and actual landslides"""
        if self.optimal_n is None or self.optimal_alpha is None:
            print("Please run grid search optimization first")
            return
        
        # Calculate effective rainfall and threshold with optimal parameters
        effective_rainfall = self.calculate_effective_rainfall(self.optimal_n, self.optimal_alpha)
        duration = np.full_like(effective_rainfall, self.optimal_n)
        thresholds = self.calculate_effective_rainfall(self.optimal_n, self.optimal_alpha)
        
        # Get actual landslide data
        actual_landslide = self.data[self.landslide_column].values
        dates = self.data.index
        
        # Create plots
        plt.figure(figsize=(15, 8))
        
        # Plot effective rainfall
        plt.subplot(2, 1, 1)
        plt.plot(dates, effective_rainfall, label='Effective Rainfall')
        plt.scatter(dates[actual_landslide == 1], 
                   effective_rainfall[actual_landslide == 1], 
                   color='red', marker='*', s=100, label='Landslide Occurrence')
        plt.title(f'Effective Rainfall and Landslide Events with Optimal Parameters (n={self.optimal_n}, α={self.optimal_alpha:.2f})')
        plt.ylabel('Effective Rainfall (mm)')
        plt.legend()
        plt.grid(True)
        
        # Plot threshold vs actual landslides
        plt.subplot(2, 1, 2)
        plt.plot(dates, thresholds, label='Critical Threshold')
        plt.scatter(dates[actual_landslide == 1], 
                   thresholds[actual_landslide == 1], 
                   color='red', marker='*', s=100, label='Landslide Occurrence')
        plt.title(f'Critical Rainfall Threshold and Landslide Events with Optimal Parameters (n={self.optimal_n}, α={self.optimal_alpha:.2f})')
        plt.xlabel('Date')
        plt.ylabel('Critical Threshold')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        # Use os.path to ensure cross-platform compatibility
        output_path = os.path.join(os.getcwd(), 'threshold_landslide_relationship.png')
        plt.savefig(output_path, dpi=300)
        plt.show()
    
    def export_results(self, filename='parameter_optimization_results.csv'):
        """Export optimization results to CSV file"""
        if not self.results:
            print("Please run grid search optimization first")
            return
        
        results_df = pd.DataFrame(self.results)
        # Use os.path to ensure cross-platform compatibility
        output_path = os.path.join(os.getcwd(), filename)
        results_df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example Excel file path
    file_path = "pingjiang_rainfall_landslide_data.xlsx"
    
    # Create model instance
    model = RainfallLandslideModel(file_path)
    
    # Load data
    model.load_data(date_column='date', rainfall_column='rainfall', landslide_column='landslide')
    
    # Perform grid search optimization
    results = model.grid_search_optimization()
    
    # Plot results heatmap
    model.plot_results_heatmap()
    
    # Plot threshold vs landslide relationship
    model.plot_optimal_threshold()
    
    # Export results
    model.export_results()    