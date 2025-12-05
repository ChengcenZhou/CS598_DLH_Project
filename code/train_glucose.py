import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

class SimpleRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_pytorch_model(model, X_train, y_train, epochs=50):
    """Train PyTorch model"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
    y_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
    
    return model

def evaluate_pytorch_model(model, X_test, y_test):
    """Evaluate PyTorch model"""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).unsqueeze(-1)
        predictions = model(X_tensor).numpy().flatten()
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return rmse

def run_glucose_experiment(data_path):
    """Run complete glucose forecasting experiment"""
    print("Loading OHIO data...")
    train_data, test_data = load_ohio_data(data_path)
    
    # Create sequences
    X_train_seq, y_train = create_sequences(train_data, history_len=12, horizon=6)
    X_test_seq, y_test = create_sequences(test_data, history_len=12, horizon=6)
    
    # Extract features for non-sequential models
    X_train_feat = extract_additional_features(X_train_seq)
    X_test_feat = extract_additional_features(X_test_seq)
    
    # Initialize simulator
    simulator = GlucoseNonstationaritySimulator(train_data['glucose'].values)
    
    all_results = []
    
    simulation_methods = [
        ('Raw Simulated', lambda n: simulator.raw_simulated(n)),
        ('MeanShift-Constant', lambda n: simulator.meanshift_constant(n)),
        ('MeanSDShift-Constant', lambda n: simulator.meansdshift_constant(n)),
        ('MeanShift-Varying', lambda n: simulator.meanshift_varying(n)),
        ('MeanSDShift-Varying', lambda n: simulator.meansdshift_varying(n)),
        ('Our Method', lambda n: simulator.learned_nonstationarity(n))
    ]
    
    for method_name, sim_func in simulation_methods:
        print(f"\n{method_name}...")
        
        # Generate simulated data
        sim_glucose = sim_func(len(train_data))
        sim_df = pd.DataFrame({'glucose': sim_glucose})
        X_sim_seq, y_sim = create_sequences(sim_df, history_len=12, horizon=6)
        X_sim_feat = extract_additional_features(X_sim_seq)
        
        results = {'dataset': method_name}
        
        # 1. Linear Regression
        lr = LinearRegression()
        lr.fit(X_sim_feat, y_sim)
        y_pred = lr.predict(X_test_feat)
        results['REG'] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        
        # 2. Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_sim_feat, y_sim)
        y_pred = rf.predict(X_test_feat)
        results['RF'] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
        
        # 3. RNN
        rnn_model = SimpleRNN()
        rnn_model = train_pytorch_model(rnn_model, X_sim_seq, y_sim)
        results['RNN'] = round(evaluate_pytorch_model(rnn_model, X_test_seq, y_test), 2)
        
        # 4. LSTM
        lstm_model = SimpleLSTM()
        lstm_model = train_pytorch_model(lstm_model, X_sim_seq, y_sim)
        results['LSTM'] = round(evaluate_pytorch_model(lstm_model, X_test_seq, y_test), 2)
        
        all_results.append(results)
        print(f"  REG: {results['REG']}, RF: {results['RF']}, RNN: {results['RNN']}, LSTM: {results['LSTM']}")
    
    # Real data baseline
    print("\nReal Data...")
    results = {'dataset': 'Real Data'}
    
    lr = LinearRegression()
    lr.fit(X_train_feat, y_train)
    y_pred = lr.predict(X_test_feat)
    results['REG'] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train_feat, y_train)
    y_pred = rf.predict(X_test_feat)
    results['RF'] = round(np.sqrt(mean_squared_error(y_test, y_pred)), 2)
    
    rnn_model = SimpleRNN()
    rnn_model = train_pytorch_model(rnn_model, X_train_seq, y_train)
    results['RNN'] = round(evaluate_pytorch_model(rnn_model, X_test_seq, y_test), 2)
    
    lstm_model = SimpleLSTM()
    lstm_model = train_pytorch_model(lstm_model, X_train_seq, y_train)
    results['LSTM'] = round(evaluate_pytorch_model(lstm_model, X_test_seq, y_test), 2)
    
    all_results.append(results)
    print(f"  REG: {results['REG']}, RF: {results['RF']}, RNN: {results['RNN']}, LSTM: {results['LSTM']}")
    
    # Save results
    with open('glucose_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

if __name__ == "__main__":
    results = run_glucose_experiment("path/to/OHIO")
    print("\nFinal Results:")
    for r in results:
        print(r)
