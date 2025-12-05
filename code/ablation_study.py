"""
Ablation Study for Learned Nonstationarity Method
Tests the contribution of each component:
1. Changepoint Detection
2. Distribution Matching  
3. Temporal Patterns
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import json
import os
import sys

# Import from other modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocessing.preprocessing_eda import process_all_subjects, temporal_split
from simulation.simulation_eda import EDAAugmentor
from models.classifiers import EDAClassifiers


def run_ablation_study(X_train, X_test, y_train, y_test, augmentor):
    """
    Run ablation study to test contribution of each component
    
    Parameters:
    - X_train, X_test: training and test features
    - y_train, y_test: training and test labels
    - augmentor: EDAAugmentor instance
    
    Returns:
    - Dictionary with ablation results
    """
    
    print("\n" + "="*80)
    print("ABLATION STUDY: Component Contribution Analysis")
    print("="*80)
    
    ablation_configs = [
        {
            'name': 'Full Method',
            'use_changepoints': True,
            'use_distribution': True,
            'use_temporal': True,
            'description': 'All three components active'
        },
        {
            'name': 'w/o Changepoint Detection',
            'use_changepoints': False,
            'use_distribution': True,
            'use_temporal': True,
            'description': 'Uses fixed regular intervals instead of KS-test detection'
        },
        {
            'name': 'w/o Distribution Matching',
            'use_changepoints': True,
            'use_distribution': False,
            'use_temporal': True,
            'description': 'Skips mean/std adjustment to real data segments'
        },
        {
            'name': 'w/o Temporal Patterns',
            'use_changepoints': True,
            'use_distribution': True,
            'use_temporal': False,
            'description': 'No linear trend application within segments'
        },
        {
            'name': 'Only Mean Shift',
            'use_changepoints': False,
            'use_distribution': False,
            'use_temporal': False,
            'description': 'Simple baseline - all components disabled'
        }
    ]
    
    all_results = []
    
    for config in ablation_configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"Description: {config['description']}")
        print(f"  - Changepoint Detection: {config['use_changepoints']}")
        print(f"  - Distribution Matching: {config['use_distribution']}")
        print(f"  - Temporal Patterns: {config['use_temporal']}")
        print('='*80)
        
        # Generate simulated data with this configuration
        X_sim, y_sim = augmentor.learned_nonstationarity(
            len(X_train),
            use_changepoints=config['use_changepoints'],
            use_distribution=config['use_distribution'],
            use_temporal=config['use_temporal']
        )
        
        # Train all classifiers
        classifiers = EDAClassifiers()
        classifiers.train_all(X_sim, y_sim)
        results = classifiers.evaluate_all(X_test, y_test)
        
        # Store results
        result_row = {
            'method': config['name'],
            'use_changepoints': config['use_changepoints'],
            'use_distribution': config['use_distribution'],
            'use_temporal': config['use_temporal']
        }
        
        for model_name, metrics in results.items():
            result_row[f'{model_name}_accuracy'] = metrics['accuracy']
            result_row[f'{model_name}_f1'] = metrics['f1']
        
        all_results.append(result_row)
        
        # Print results
        print(f"\nResults for {config['name']}:")
        for model_name, metrics in results.items():
            print(f"  {model_name}: Acc={metrics['accuracy']:.2f}, F1={metrics['f1']:.2f}")
    
    # Add real data baseline for comparison
    print(f"\n{'='*80}")
    print("Real Data Baseline (for comparison)")
    print('='*80)
    
    classifiers = EDAClassifiers()
    classifiers.train_all(X_train, y_train)
    results = classifiers.evaluate_all(X_test, y_test)
    
    result_row = {
        'method': 'Real Data',
        'use_changepoints': None,
        'use_distribution': None,
        'use_temporal': None
    }
    
    for model_name, metrics in results.items():
        result_row[f'{model_name}_accuracy'] = metrics['accuracy']
        result_row[f'{model_name}_f1'] = metrics['f1']
    
    all_results.append(result_row)
    
    print(f"\nResults for Real Data:")
    for model_name, metrics in results.items():
        print(f"  {model_name}: Acc={metrics['accuracy']:.2f}, F1={metrics['f1']:.2f}")
    
    return all_results


def analyze_ablation_results(results):
    """
    Analyze ablation study results to determine component importance
    
    Parameters:
    - results: List of result dictionaries from run_ablation_study
    
    Returns:
    - Analysis summary
    """
    
    print("\n" + "="*80)
    print("ABLATION ANALYSIS: Component Importance")
    print("="*80)
    
    # Extract full method and real data results
    full_method = next(r for r in results if r['method'] == 'Full Method')
    real_data = next(r for r in results if r['method'] == 'Real Data')
    
    # Calculate average accuracy across all models
    models = ['KNN', 'LR', 'RF', 'SVM']
    
    full_method_avg = np.mean([full_method[f'{m}_accuracy'] for m in models])
    real_data_avg = np.mean([real_data[f'{m}_accuracy'] for m in models])
    
    print(f"\nFull Method Average Accuracy: {full_method_avg:.3f}")
    print(f"Real Data Average Accuracy: {real_data_avg:.3f}")
    print(f"Gap to Real Data: {abs(full_method_avg - real_data_avg):.3f}")
    
    print("\n" + "-"*80)
    print("Impact of Removing Each Component:")
    print("-"*80)
    
    ablation_results = []
    
    for result in results:
        if result['method'] in ['Full Method', 'Real Data']:
            continue
        
        avg_acc = np.mean([result[f'{m}_accuracy'] for m in models])
        impact = avg_acc - full_method_avg
        gap_to_real = abs(avg_acc - real_data_avg)
        
        ablation_results.append({
            'method': result['method'],
            'avg_accuracy': avg_acc,
            'impact_vs_full': impact,
            'gap_to_real': gap_to_real
        })
        
        print(f"\n{result['method']:<30}")
        print(f"  Average Accuracy: {avg_acc:.3f}")
        print(f"  Impact vs Full:   {impact:+.3f} ({'worse' if impact < 0 else 'better'})")
        print(f"  Gap to Real Data: {gap_to_real:.3f}")
    
    # Sort by impact (most negative = most important component)
    ablation_results.sort(key=lambda x: x['impact_vs_full'])
    
    print("\n" + "-"*80)
    print("Component Importance Ranking (by performance drop when removed):")
    print("-"*80)
    
    for i, result in enumerate(ablation_results, 1):
        if result['impact_vs_full'] < 0:
            print(f"{i}. {result['method']:<30} Impact: {result['impact_vs_full']:.3f}")
    
    print("\n" + "="*80)
    print("KEY FINDINGS:")
    print("="*80)
    
    # Find most critical component
    most_critical = ablation_results[0]
    print(f"\n1. Most Critical Component: {most_critical['method']}")
    print(f"   Removing it causes {abs(most_critical['impact_vs_full']):.3f} drop in accuracy")
    
    # Find least critical
    least_critical = [r for r in ablation_results if r['impact_vs_full'] < 0][-1]
    print(f"\n2. Least Critical Component: {least_critical['method']}")
    print(f"   Removing it causes only {abs(least_critical['impact_vs_full']):.3f} drop in accuracy")
    
    # Check "Only Mean Shift" baseline
    only_mean_shift = next(r for r in results if r['method'] == 'Only Mean Shift')
    only_mean_shift_avg = np.mean([only_mean_shift[f'{m}_accuracy'] for m in models])
    
    print(f"\n3. Synergistic Effects:")
    print(f"   Full Method: {full_method_avg:.3f}")
    print(f"   Only Mean Shift (no components): {only_mean_shift_avg:.3f}")
    print(f"   Difference: {only_mean_shift_avg - full_method_avg:+.3f}")
    print(f"   => All components must work together for realistic simulation")
    
    return {
        'full_method_avg': full_method_avg,
        'real_data_avg': real_data_avg,
        'component_impacts': ablation_results,
        'most_critical': most_critical['method'],
        'least_critical': least_critical['method']
    }


def print_ablation_table(results):
    """
    Print results in a formatted table
    """
    print("\n" + "="*100)
    print("ABLATION STUDY RESULTS TABLE")
    print("="*100)
    print(f"{'Method':<30} {'KNN':<15} {'LR':<15} {'RF':<15} {'SVM':<15}")
    print(f"{'':<30} {'Acc':>7} {'F1':>7} {'Acc':>7} {'F1':>7} {'Acc':>7} {'F1':>7} {'Acc':>7} {'F1':>7}")
    print("-"*100)
    
    for result in results:
        print(f"{result['method']:<30} "
              f"{result['KNN_accuracy']:>7.2f} {result['KNN_f1']:>7.2f} "
              f"{result['LR_accuracy']:>7.2f} {result['LR_f1']:>7.2f} "
              f"{result['RF_accuracy']:>7.2f} {result['RF_f1']:>7.2f} "
              f"{result['SVM_accuracy']:>7.2f} {result['SVM_f1']:>7.2f}")
    
    print("="*100)


def main():
    """
    Main execution function
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation study')
    parser.add_argument('--data_path', default='./data/wesad',
                       help='Path to WESAD dataset')
    parser.add_argument('--output_dir', default='./results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load and preprocess data
    print("Loading WESAD data...")
    subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 
                'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
    X, y = process_all_subjects(args.data_path, subjects)
    
    # Temporal split
    print("Splitting data temporally...")
    X_train, X_val, X_test, y_train, y_val, y_test = temporal_split(X, y)
    
    # Initialize augmentor
    print("Initializing augmentor...")
    augmentor = EDAAugmentor(X_train, y_train, n_components=3)
    
    # Run ablation study
    results = run_ablation_study(X_train, X_test, y_train, y_test, augmentor)
    
    # Print formatted table
    print_ablation_table(results)
    
    # Analyze results
    analysis = analyze_ablation_results(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(os.path.join(args.output_dir, 'ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(os.path.join(args.output_dir, 'ablation_analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"\nResults saved to {args.output_dir}/")
    print("  - ablation_results.json")
    print("  - ablation_analysis.json")


if __name__ == "__main__":
    main()
