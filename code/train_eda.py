from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import json

def train_and_evaluate_all(X_train, X_test, y_train, y_test, dataset_name):
    """Train and evaluate all models"""
    models = {
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'LR': LogisticRegression(max_iter=1000),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf')
    }
    
    results = {'dataset': dataset_name}
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='binary')
        
        results[f'{name}_accuracy'] = round(acc, 2)
        results[f'{name}_f1'] = round(f1, 2)
        
        print(f"{name}: Accuracy={acc:.2f}, F1={f1:.2f}")
    
    return results

def run_eda_experiment(data_path):
    """Run complete EDA experiment"""
    print("Loading WESAD data...")
    X, y = load_wesad_data(data_path)
    
    print(f"Data shape: {X.shape}, Labels: {np.bincount(y)}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data_by_time(X, y)
    
    # Initialize simulator
    simulator = NonstationaritySimulator(X_train, y_train)
    
    all_results = []
    
    # 1. Raw Simulated
    print("\n1. Raw Simulated...")
    X_sim, y_sim = simulator.raw_simulated(len(X_train))
    results = train_and_evaluate_all(X_sim, X_test, y_sim, y_test, "Raw Simulated")
    all_results.append(results)
    
    # 2. MeanShift-Constant
    print("\n2. MeanShift-Constant...")
    X_sim, y_sim = simulator.meanshift_constant(len(X_train))
    results = train_and_evaluate_all(X_sim, X_test, y_sim, y_test, "MeanShift-Constant")
    all_results.append(results)
    
    # 3. MeanSDShift-Constant
    print("\n3. MeanSDShift-Constant...")
    X_sim, y_sim = simulator.meansdshift_constant(len(X_train))
    results = train_and_evaluate_all(X_sim, X_test, y_sim, y_test, "MeanSDShift-Constant")
    all_results.append(results)
    
    # 4. MeanShift-Varying
    print("\n4. MeanShift-Varying...")
    X_sim, y_sim = simulator.meanshift_varying(len(X_train))
    results = train_and_evaluate_all(X_sim, X_test, y_sim, y_test, "MeanShift-Varying")
    all_results.append(results)
    
    # 5. MeanSDShift-Varying
    print("\n5. MeanSDShift-Varying...")
    X_sim, y_sim = simulator.meansdshift_varying(len(X_train))
    results = train_and_evaluate_all(X_sim, X_test, y_sim, y_test, "MeanSDShift-Varying")
    all_results.append(results)
    
    # 6. Our Method
    print("\n6. Our Method (Learned Nonstationarity)...")
    X_sim, y_sim = simulator.learned_nonstationarity(len(X_train))
    results = train_and_evaluate_all(X_sim, X_test, y_sim, y_test, "Our Method")
    all_results.append(results)
    
    # 7. Real Data (baseline)
    print("\n7. Real Data...")
    results = train_and_evaluate_all(X_train, X_test, y_train, y_test, "Real Data")
    all_results.append(results)
    
    # Save results
    with open('eda_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    results = run_eda_experiment("path/to/WESAD")
    print("\nFinal Results:")
    print(json.dumps(results, indent=2))
