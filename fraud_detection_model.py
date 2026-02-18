# fraud_detection_model.py
"""
Fraud Detection Model Training Module
Implements various techniques to handle imbalanced data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_recall_curve, auc, f1_score,
                           precision_score, recall_score, roc_auc_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """Comprehensive fraud detection model with imbalance handling"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def load_data(self, filepath='data/fraud_detection_dataset.csv'):
        """Load preprocessed dataset"""
        self.df = pd.read_csv(filepath)
        print(f"✅ Data loaded: {self.df.shape}")
        
        # Prepare features and target
        self.X = self.df.drop('Class', axis=1)
        self.y = self.df['Class']
        
        print(f"Features: {self.X.shape[1]}")
        print(f"Target distribution:\n{self.y.value_counts()}")
        
    def prepare_data(self, test_size=0.2):
        """Split and scale data"""
        # Split first to avoid data leakage
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, 
            random_state=self.random_state, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"\n📊 Data Split:")
        print(f"Training set: {self.X_train.shape[0]:,} samples")
        print(f"Test set: {self.X_test.shape[0]:,} samples")
        print(f"\nTraining set distribution:")
        print(pd.Series(self.y_train).value_counts(normalize=True))
        
    def train_baseline(self):
        """Train baseline model without balancing"""
        print("\n" + "="*60)
        print("🔵 TRAINING BASELINE MODEL (NO BALANCING)")
        print("="*60)
        
        model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=self.random_state)
        # model.fit(self.X_train_scaled, self.y_train) -> Logic remains same, just updating instantiation lines below
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        self._evaluate_model('Baseline (No Balancing)', y_pred, y_proba, model)
        self.models['Baseline'] = model
        
    def train_with_smote(self):
        """Train with SMOTE oversampling"""
        print("\n" + "="*60)
        print("🟢 TRAINING WITH SMOTE (OVERSAMPLING)")
        print("="*60)
        
        # Apply SMOTE
        smote = SMOTE(random_state=self.random_state)
        X_train_smote, y_train_smote = smote.fit_resample(self.X_train_scaled, self.y_train)
        
        print(f"After SMOTE - Class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=self.random_state)
        # model.fit(self.X_train_scaled, self.y_train) -> Logic remains same, just updating instantiation lines below
        model.fit(X_train_smote, y_train_smote)
        
        # Predict
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        self._evaluate_model('SMOTE Oversampling', y_pred, y_proba, model)
        self.models['SMOTE'] = model
        
    def train_with_undersampling(self):
        """Train with random undersampling"""
        print("\n" + "="*60)
        print("🟡 TRAINING WITH RANDOM UNDERSAMPLING")
        print("="*60)
        
        # Apply undersampling
        undersampler = RandomUnderSampler(random_state=self.random_state)
        X_train_under, y_train_under = undersampler.fit_resample(self.X_train_scaled, self.y_train)
        
        print(f"After Undersampling - Class distribution: {pd.Series(y_train_under).value_counts().to_dict()}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=50, max_depth=20, random_state=self.random_state)
        # model.fit(self.X_train_scaled, self.y_train) -> Logic remains same, just updating instantiation lines below
        model.fit(X_train_under, y_train_under)
        
        # Predict
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        self._evaluate_model('Random Undersampling', y_pred, y_proba, model)
        self.models['Undersampling'] = model
        
    def train_with_class_weight(self):
        """Train with class weight adjustment"""
        print("\n" + "="*60)
        print("🟣 TRAINING WITH CLASS WEIGHT ADJUSTMENT")
        print("="*60)
        
        # Calculate class weights automatically
        model = RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=self.random_state
        )
        model.fit(self.X_train_scaled, self.y_train)
        
        # Predict
        y_pred = model.predict(self.X_test_scaled)
        y_proba = model.predict_proba(self.X_test_scaled)[:, 1]
        
        # Evaluate
        self._evaluate_model('Class Weight Balancing', y_pred, y_proba, model)
        self.models['Class Weight'] = model
        
    def train_isolation_forest(self):
        """Train Isolation Forest for anomaly detection"""
        print("\n" + "="*60)
        print("🔴 TRAINING ISOLATION FOREST (ANOMALY DETECTION)")
        print("="*60)
        
        # Isolation Forest expects contamination rate
        contamination = self.y.mean()  # Use actual fraud rate
        model = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Fit on training data
        model.fit(self.X_train_scaled)
        
        # Predict (-1 for anomalies/fraud, 1 for normal)
        y_pred = model.predict(self.X_test_scaled)
        # Convert: -1 -> 1 (fraud), 1 -> 0 (normal)
        y_pred = np.where(y_pred == -1, 1, 0)
        
        # No probability for Isolation Forest
        y_proba = None
        
        # Evaluate
        self._evaluate_model('Isolation Forest', y_pred, y_proba, model, is_anomaly=True)
        self.models['Isolation Forest'] = model
        
    def _evaluate_model(self, name, y_pred, y_proba, model, is_anomaly=False):
        """Evaluate and store model results"""
        
        # Calculate metrics
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        
        # ROC-AUC (if probabilities available)
        roc_auc = None
        if y_proba is not None:
            roc_auc = roc_auc_score(self.y_test, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Store results
        self.results[name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'model': model
        }
        
        # Print results
        print(f"\n📊 Results for {name}:")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        if roc_auc:
            print(f"   ROC-AUC: {roc_auc:.4f}")
        
        print(f"\n   Confusion Matrix:")
        print(f"   TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"   FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        
        # Calculate fraud detection rate
        fraud_caught = cm[1,1]
        total_fraud = cm[1,0] + cm[1,1]
        detection_rate = fraud_caught / total_fraud if total_fraud > 0 else 0
        print(f"   Fraud Detection Rate: {detection_rate:.2%}")
        
    def compare_models(self, save_path='reports/figures/'):
        """Compare all models visually"""
        import os
        os.makedirs(save_path, exist_ok=True)
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Fraud Detection Models Comparison', fontsize=16, fontweight='bold')
        
        # 1. Bar plot of metrics
        ax1 = axes[0, 0]
        models = list(self.results.keys())
        x = np.arange(len(models))
        width = 0.25
        
        precision_scores = [self.results[m]['precision'] for m in models]
        recall_scores = [self.results[m]['recall'] for m in models]
        f1_scores = [self.results[m]['f1_score'] for m in models]
        
        ax1.bar(x - width, precision_scores, width, label='Precision', color='#3498db')
        ax1.bar(x, recall_scores, width, label='Recall', color='#2ecc71')
        ax1.bar(x + width, f1_scores, width, label='F1-Score', color='#e74c3c')
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Performance Metrics by Model', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim([0, 1])
        
        # 2. Fraud Detection Rate
        ax2 = axes[0, 1]
        fraud_rates = []
        for name, result in self.results.items():
            cm = result['confusion_matrix']
            rate = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            fraud_rates.append(rate)
        
        colors = ['#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#e74c3c']
        bars = ax2.bar(models, fraud_rates, color=colors[:len(models)])
        ax2.set_title('Fraud Detection Rate by Model', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Detection Rate')
        ax2.set_ylim([0, 1])
        ax2.set_xticklabels(models, rotation=45, ha='right')
        
        # Add value labels
        for bar, rate in zip(bars, fraud_rates):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # 3. Confusion Matrix Heatmap for best model
        ax3 = axes[1, 0]
        # Find best model by F1-score
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])[0]
        best_cm = self.results[best_model]['confusion_matrix']
        
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
                   xticklabels=['Predicted Normal', 'Predicted Fraud'],
                   yticklabels=['Actual Normal', 'Actual Fraud'])
        ax3.set_title(f'Best Model: {best_model}\nConfusion Matrix', fontsize=12, fontweight='bold')
        
        # 4. Metrics comparison table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create comparison table
        table_data = []
        for model in models:
            row = [
                model,
                f"{self.results[model]['precision']:.3f}",
                f"{self.results[model]['recall']:.3f}",
                f"{self.results[model]['f1_score']:.3f}"
            ]
            table_data.append(row)
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Model', 'Precision', 'Recall', 'F1-Score'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        
        plt.tight_layout()
        
        # Save
        plt.savefig(f'{save_path}model_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}model_comparison.pdf', bbox_inches='tight')
        print(f"\n✅ Comparison plots saved to {save_path}")
        
        plt.close(fig)  # Close instead of showing interactively
        
        # Print summary
        self._print_summary_table()
        
    def _print_summary_table(self):
        """Print summary comparison table"""
        print("\n" + "="*70)
        print("📊 MODEL COMPARISON SUMMARY")
        print("="*70)
        print(f"{'Model':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Detect Rate':<12}")
        print("-"*70)
        
        for name, results in self.results.items():
            cm = results['confusion_matrix']
            detect_rate = cm[1,1] / (cm[1,0] + cm[1,1]) if (cm[1,0] + cm[1,1]) > 0 else 0
            print(f"{name:<25} {results['precision']:<12.4f} {results['recall']:<12.4f} "
                  f"{results['f1_score']:<12.4f} {detect_rate:<12.2%}")
        
        # Find best model for different metrics
        best_f1 = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        best_recall = max(self.results.items(), key=lambda x: x[1]['recall'])
        
        print("\n" + "="*70)
        print("🏆 RECOMMENDATIONS:")
        print(f"• Best overall (F1-Score): {best_f1[0]} ({best_f1[1]['f1_score']:.4f})")
        print(f"• Best fraud capture (Recall): {best_recall[0]} ({best_recall[1]['recall']:.4f})")
        print("="*70)

def main():
    """Main execution"""
    print("🚀 Starting Fraud Detection Model Training")
    print("="*60)
    
    # Initialize model trainer
    trainer = FraudDetectionModel(random_state=42)
    
    # Load data
    trainer.load_data()
    
    # Prepare data
    trainer.prepare_data(test_size=0.2)
    
    # Train all models
    trainer.train_baseline()
    trainer.train_with_smote()
    trainer.train_with_undersampling()
    trainer.train_with_class_weight()
    trainer.train_isolation_forest()
    
    # Compare results
    trainer.compare_models()
    
    return trainer

if __name__ == "__main__":
    trainer = main()
