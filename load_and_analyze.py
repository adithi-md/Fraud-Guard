# load_and_analyze.py
"""
Fraud Detection - Data Loading and Analysis Module
This script handles data loading, exploration, and visualization of imbalanced datasets
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

class FraudDataLoader:
    """Handle loading and creating fraud detection datasets"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.data = None
        self.target = None
        
    def create_synthetic_dataset(self, n_samples=20000, fraud_ratio=0.01):
        """
        Create synthetic transaction data with fraud cases
        This mimics real-world credit card fraud patterns
        """
        print(f"🔨 Creating synthetic dataset with {fraud_ratio*100}% fraud ratio...")
        
        # Create imbalanced dataset
        # Optimizing for speed (20k samples) and accuracy (better separation)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=25,  # 25 features like transaction amount, time, location etc
            n_informative=22, # Increased for better pattern recognition
            n_redundant=3,
            n_clusters_per_class=1,
            weights=[1-fraud_ratio, fraud_ratio],  # Imbalanced classes
            flip_y=0.01,  # Slight noise for realism
            class_sep=1.5,  # Better separation between classes for higher accuracy
            random_state=self.random_state
        )
        
        # Convert to DataFrame with meaningful column names
        feature_cols = [f'V{i}' for i in range(1, 26)]  # Like credit card dataset
        df = pd.DataFrame(X, columns=feature_cols)
        df['Class'] = y
        
        # Add realistic transaction features
        df['Amount'] = np.where(
            df['Class'] == 1,
            np.random.uniform(500, 5000, len(df)),  # Fraud: higher amounts
            np.random.uniform(1, 1000, len(df))     # Normal: lower amounts
        )
        
        # Add time feature (30 days of transactions)
        df['Time'] = np.random.randint(0, 2592000, n_samples)  # Seconds in 30 days
        
        self.data = df
        self.target = 'Class'
        
        print(f"✅ Dataset created: {df.shape[0]:,} transactions, {df.shape[1]-1} features")
        return df
    
    def load_real_dataset(self):
        """
        Attempt to load real credit card fraud dataset
        You'll need to download from Kaggle first
        """
        try:
            # Try loading the Kaggle Credit Card Fraud dataset
            df = pd.read_csv('creditcard.csv')
            print("✅ Loaded Credit Card Fraud Detection dataset")
            self.data = df
            self.target = 'Class'
            return df
        except FileNotFoundError:
            print("⚠️  Real dataset not found. Using synthetic data instead.")
            return self.create_synthetic_dataset()

class DataAnalyzer:
    """Analyze and visualize fraud detection datasets"""
    
    def __init__(self, df, target_col='Class'):
        self.df = df
        self.target_col = target_col
        
    def basic_info(self):
        """Display basic dataset information"""
        print("\n" + "="*50)
        print("📊 DATASET INFORMATION")
        print("="*50)
        
        print(f"\n🔹 Shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
        print(f"\n🔹 Columns:\n{self.df.columns.tolist()}")
        
        print("\n🔹 Data Types:")
        print(self.df.dtypes.value_counts())
        
        print("\n🔹 Missing Values:")
        missing = self.df.isnull().sum().sum()
        print(f"Total missing values: {missing}")
        
        if missing > 0:
            print(self.df.isnull().sum()[self.df.isnull().sum() > 0])
    
    def analyze_class_distribution(self):
        """Analyze and visualize class imbalance"""
        print("\n" + "="*50)
        print("📊 CLASS DISTRIBUTION ANALYSIS")
        print("="*50)
        
        # Calculate distribution
        class_counts = self.df[self.target_col].value_counts()
        class_percentages = self.df[self.target_col].value_counts(normalize=True) * 100
        
        # Create distribution DataFrame
        dist_df = pd.DataFrame({
            'Class': ['Normal (0)', 'Fraud (1)'],
            'Count': [class_counts.get(0, 0), class_counts.get(1, 0)],
            'Percentage': [class_percentages.get(0, 0), class_percentages.get(1, 0)]
        })
        
        print("\n🔹 Class Distribution:")
        print(dist_df.to_string(index=False))
        
        # Calculate imbalance ratio
        if class_counts.get(0, 0) > 0 and class_counts.get(1, 0) > 0:
            imbalance_ratio = class_counts[0] / class_counts[1]
            print(f"\n🔹 Imbalance Ratio (Normal:Fraud): {imbalance_ratio:.2f}:1")
            print(f"🔹 Fraud Percentage: {class_percentages.get(1, 0):.4f}%")
            
            # Accuracy trap demonstration
            dummy_accuracy = class_percentages.get(0, 0) / 100
            print(f"\n⚠️  THE ACCURACY TRAP:")
            print(f"   If model always predicts 'Normal', accuracy = {dummy_accuracy:.4f} ({dummy_accuracy*100:.2f}%)")
            print(f"   But it would catch 0% of fraud cases!")
        
        return dist_df
    
    def plot_distributions(self, save_path='reports/figures/'):
        """Create visualizations of class distribution"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Get class percentages for the pie chart
        class_counts = self.df[self.target_col].value_counts()
        class_percentages = self.df[self.target_col].value_counts(normalize=True) * 100
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fraud Detection Dataset Analysis', fontsize=16, fontweight='bold')
        
        # 1. Bar plot of class distribution
        ax1 = axes[0, 0]
        colors = ['#2ecc71', '#e74c3c']
        bars = ax1.bar(['Normal (0)', 'Fraud (1)'], class_counts.values, color=colors)
        ax1.set_title('Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Transactions')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}', ha='center', va='bottom')
        
        # 2. Pie chart
        ax2 = axes[0, 1]
        wedges, texts, autotexts = ax2.pie(class_counts.values, 
                                           labels=['Normal', 'Fraud'],
                                           autopct='%1.4f%%',
                                           colors=colors,
                                           explode=(0, 0.1))
        ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        # 3. Transaction amount distribution by class
        ax3 = axes[1, 0]
        if 'Amount' in self.df.columns:
            for cls, color, label in zip([0, 1], colors, ['Normal', 'Fraud']):
                subset = self.df[self.df[self.target_col] == cls]
                ax3.hist(subset['Amount'], bins=50, alpha=0.7, 
                        label=label, color=color, density=True)
            ax3.set_title('Transaction Amount Distribution by Class', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Transaction Amount')
            ax3.set_ylabel('Density')
            ax3.legend()
        
        # 4. Imbalance ratio text
        ax4 = axes[1, 1]
        ax4.axis('off')
        imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else 0
        textstr = f'IMBALANCE ANALYSIS\n\n'
        textstr += f'Normal Transactions: {class_counts[0]:,}\n'
        textstr += f'Fraud Transactions: {class_counts[1]:,}\n'
        textstr += f'Imbalance Ratio: {imbalance_ratio:.2f}:1\n\n'
        textstr += f'Fraud Percentage: {class_percentages[1]:.6f}%\n\n'
        textstr += f'⚠️ Accuracy Trap:\n'
        textstr += f'Dummy accuracy: {class_percentages[0]/100:.4f}'
        
        ax4.text(0.1, 0.5, textstr, transform=ax4.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plots
        plt.savefig(f'{save_path}class_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}class_distribution_analysis.pdf', bbox_inches='tight')
        print(f"\n✅ Visualizations saved to {save_path}")
        
        plt.close(fig)  # Close figure instead of showing interactively
        return fig
    
    def statistical_summary(self):
        """Generate statistical summaries for both classes"""
        print("\n" + "="*50)
        print("📊 STATISTICAL SUMMARY BY CLASS")
        print("="*50)
        
        # Separate classes
        fraud = self.df[self.df[self.target_col] == 1]
        normal = self.df[self.df[self.target_col] == 0]
        
        print(f"\n🔹 Normal Transactions (Class 0): {len(normal):,} samples")
        print(f"🔹 Fraud Transactions (Class 1): {len(fraud):,} samples")
        
        # Numerical columns summary
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if 'Amount' in numeric_cols:
            print("\n🔹 Transaction Amount Statistics:")
            amount_stats = pd.DataFrame({
                'Normal': normal['Amount'].describe(),
                'Fraud': fraud['Amount'].describe()
            })
            print(amount_stats.round(2))
        
        return {
            'normal': normal.describe(),
            'fraud': fraud.describe()
        }

def main():
    """Main execution function"""
    print("🚀 Starting Fraud Detection Data Analysis")
    print("="*60)
    
    # Initialize loader
    loader = FraudDataLoader(random_state=42)
    
    # Create dataset (use synthetic for demo, switch to real if available)
    df = loader.create_synthetic_dataset(n_samples=100000, fraud_ratio=0.01)
    
    # Initialize analyzer
    analyzer = DataAnalyzer(df, target_col='Class')
    
    # Run analysis
    analyzer.basic_info()
    dist_df = analyzer.analyze_class_distribution()
    stats = analyzer.statistical_summary()
    
    # Create visualizations
    analyzer.plot_distributions()
    
    # Save processed data
    df.to_csv('data/fraud_detection_dataset.csv', index=False)
    print("\n✅ Dataset saved to 'data/fraud_detection_dataset.csv'")
    
    return df, analyzer

if __name__ == "__main__":
    df, analyzer = main()
