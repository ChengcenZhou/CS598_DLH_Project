"""
Main script to run all experiments from the paper
"""

import argparse
import json
import pandas as pd
from preprocessing_eda import load_wesad_data, split_data_by_time
from preprocessing_glucose import load_ohio_data, create_sequences, extract_additional_features
from train_eda import run_eda_experiment
from train_glucose import run_glucose_experiment

def print_results_table_eda(results):
    """Print EDA results in table format"""
    print("\n" + "="*100)
    print("EDA STRESS CLASSIFICATION RESULTS")
    print("="*100)
    print(f"{'Dataset':<25} {'KNN':<15} {'LR':<15} {'RF':<15} {'SVM':<15}")
    print(f"{'':<25} {'Acc':>7} {'F1':>7} {'Acc':>7} {'F1':>7} {'Acc':>7} {'F1':>7} {'Acc':>7} {'F1':>7}")
    print("-"*100)
    
    for r in results:
        print(f"{r['dataset']:<25} {r['KNN_accuracy']:>7.2f} {r['KNN_f1']:>7.2f} "
              f"{r['LR_accuracy']:>7.2f} {r['LR_f1']:>7.2f} "
              f"{r['RF_accuracy']:>7.2f} {r['RF_f1']:>7.2f} "
              f"{r['SVM_accuracy']:>7.2f} {r['SVM_f1']:>7.2f}")
    print("="*100)

def print_results_table_glucose(results):
    """Print glucose results in table format"""
    print("\n" + "="*80)
    print("GLUCOSE FORECASTING RESULTS (RMSE in mg/dL)")
    print("="*80)
    print(f"{'Dataset':<25} {'REG':>10} {'RF':>10} {'RNN':>10} {'LSTM':>10}")
    print("-"*80)
    
    for r in results:
        print(f"{r['dataset']:<25} {r['REG']:>10.2f} {r['RF']:>10.2f} "
              f"{r['RNN']:>10.2f} {r['LSTM']:>10.2f}")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Run health time series experiments')
    parser.add_argument('--experiment', type=str, choices=['eda', 'glucose', 'both'],
                        default='both', help='Which experiment to run')
    parser.add_argument('--wesad_path', type=str, default='./wesad',
                        help='Path to WESAD dataset')
    parser.add_argument('--ohio_path', type=str, default='./ohio',
                        help='Path to OHIO-T1DM dataset')
    parser.add_argument('--output', type=str, default='results.json',
                        help='Output file for results')
    
    args = parser.parse_args()
    
    all_results = {}
    
    if args.experiment in ['eda', 'both']:
        print("\n" + "="*80)
        print("RUNNING EDA STRESS CLASSIFICATION EXPERIMENT")
        print("="*80)
        eda_results = run_eda_experiment(args.wesad_path)
        all_results['eda'] = eda_results
        print_results_table_eda(eda_results)
    
    if args.experiment in ['glucose', 'both']:
        print("\n" + "="*80)
        print("RUNNING GLUCOSE FORECASTING EXPERIMENT")
        print("="*80)
        glucose_results = run_glucose_experiment(args.ohio_path)
        all_results['glucose'] = glucose_results
        print_results_table_glucose(glucose_results)
    
    # Save all results
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
