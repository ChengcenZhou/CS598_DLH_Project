"""
Changepoint Distribution Analysis for OHIO Glucose Data
Generates plots similar to Figure 3 from the paper
"Simulation of Health Time Series with Nonstationarity"
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

def detect_changepoints_ks_test(data, window_size=12, significance=0.01):
    """
    Detect changepoints using Kolmogorov-Smirnov test
    
    Parameters:
    - data: glucose time series (numpy array)
    - window_size: number of points in each window (default 12 = 1 hour for 5-min intervals)
    - significance: p-value threshold for detecting significant changes
    
    Returns:
    - List of changepoint indices
    """
    changepoints = [0]  # Start with beginning
    
    for i in range(window_size, len(data) - window_size, window_size):
        # Get windows before and after potential changepoint
        before = data[i-window_size:i]
        after = data[i:i+window_size]
        
        # Two-sample Kolmogorov-Smirnov test
        statistic, p_value = stats.ks_2samp(before, after)
        
        if p_value < significance:
            changepoints.append(i)
    
    changepoints.append(len(data))  # End with final point
    return changepoints

def extract_changepoint_properties(data, changepoints):
    """
    Extract properties at each changepoint
    
    Parameters:
    - data: glucose time series
    - changepoints: list of changepoint indices
    
    Returns:
    - durations: time between changepoints (in minutes, assuming 5-min intervals)
    - mean_changes: change in mean glucose between consecutive segments
    - sd_changes: change in standard deviation between consecutive segments
    """
    durations = []
    mean_changes = []
    sd_changes = []
    
    for i in range(len(changepoints) - 2):
        # Current segment
        start1 = changepoints[i]
        end1 = changepoints[i+1]
        segment1 = data[start1:end1]
        
        # Next segment
        start2 = changepoints[i+1]
        end2 = changepoints[i+2]
        segment2 = data[start2:end2]
        
        # Duration between changepoints (in minutes)
        duration = (end1 - start1) * 5  # 5-minute intervals in OHIO
        durations.append(duration)
        
        # Mean and SD changes
        mean_change = np.mean(segment2) - np.mean(segment1)
        sd_change = np.std(segment2) - np.std(segment1)
        
        mean_changes.append(mean_change)
        sd_changes.append(sd_change)
    
    return np.array(durations), np.array(mean_changes), np.array(sd_changes)

def plot_distribution_with_fits(data, ax, title, xlabel, fit_distributions=['norm', 'expon', 'uniform']):
    """
    Plot histogram with fitted distributions
    
    Parameters:
    - data: array of values to plot
    - ax: matplotlib axis object
    - title: plot title
    - xlabel: x-axis label
    - fit_distributions: list of distribution names to fit
    """
    # Create histogram
    counts, bins, patches = ax.hist(data, bins=30, density=True, 
                                     alpha=0.6, color='skyblue', 
                                     edgecolor='white', linewidth=0.5)
    
    # Generate x values for smooth curves
    x_range = np.linspace(data.min(), data.max(), 200)
    
    # Color mapping
    colors = {'norm': 'blue', 'expon': 'red', 'uniform': 'orange', 'bimodal': 'green'}
    
    # Fit and plot each distribution
    best_fit = None
    best_aic = np.inf
    
    for dist_name in fit_distributions:
        try:
            if dist_name == 'norm':
                # Normal distribution
                mu, std = stats.norm.fit(data)
                pdf = stats.norm.pdf(x_range, mu, std)
                label = f'Best fit: norm' if best_fit is None else 'norm'
                
                # Calculate AIC
                log_likelihood = np.sum(stats.norm.logpdf(data, mu, std))
                aic = 2 * 2 - 2 * log_likelihood  # 2 parameters
                
            elif dist_name == 'expon':
                # Exponential distribution (shift data if needed)
                if data.min() < 0:
                    data_shifted = data - data.min()
                else:
                    data_shifted = data
                loc, scale = stats.expon.fit(data_shifted)
                pdf = stats.expon.pdf(x_range - data.min() if data.min() < 0 else x_range, loc, scale)
                label = 'expon'
                
                # Calculate AIC
                log_likelihood = np.sum(stats.expon.logpdf(data_shifted, loc, scale))
                aic = 2 * 2 - 2 * log_likelihood
                
            elif dist_name == 'uniform':
                # Uniform distribution
                a, b = data.min(), data.max()
                pdf = stats.uniform.pdf(x_range, a, b - a)
                label = 'uniform'
                
                # Calculate AIC
                log_likelihood = np.sum(stats.uniform.logpdf(data, a, b - a))
                aic = 2 * 2 - 2 * log_likelihood
            
            # Track best fit
            if aic < best_aic:
                best_aic = aic
                best_fit = dist_name
            
            # Plot
            ax.plot(x_range, pdf, color=colors[dist_name], 
                   linewidth=2, label=label, alpha=0.8)
        
        except Exception as e:
            print(f"Warning: Could not fit {dist_name}: {e}")
    
    # Formatting
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel('Probability Density', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return best_fit

def analyze_ohio_changepoints(glucose_data, plot=True, save_path=None):
    """
    Main analysis function for OHIO glucose data
    
    Parameters:
    - glucose_data: numpy array or pandas Series of glucose values
    - plot: whether to generate plots
    - save_path: path to save figure (if None, just displays)
    
    Returns:
    - Dictionary with analysis results
    """
    print("="*80)
    print("CHANGEPOINT DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Convert to numpy array if needed
    if isinstance(glucose_data, pd.Series):
        glucose_data = glucose_data.values
    
    print(f"\nData summary:")
    print(f"  Total points: {len(glucose_data)}")
    print(f"  Mean glucose: {np.mean(glucose_data):.1f} mg/dL")
    print(f"  Std glucose: {np.std(glucose_data):.1f} mg/dL")
    print(f"  Range: [{np.min(glucose_data):.1f}, {np.max(glucose_data):.1f}] mg/dL")
    
    # Detect changepoints
    print("\nDetecting changepoints using KS test...")
    changepoints = detect_changepoints_ks_test(glucose_data, window_size=12, significance=0.01)
    print(f"  Found {len(changepoints)-2} changepoints")
    
    # Extract properties
    print("\nExtracting changepoint properties...")
    durations, mean_changes, sd_changes = extract_changepoint_properties(glucose_data, changepoints)
    
    print(f"\nDuration statistics:")
    print(f"  Mean: {np.mean(durations):.1f} minutes")
    print(f"  Median: {np.median(durations):.1f} minutes")
    print(f"  Std: {np.std(durations):.1f} minutes")
    print(f"  Range: [{np.min(durations):.1f}, {np.max(durations):.1f}] minutes")
    
    print(f"\nMean change statistics:")
    print(f"  Mean: {np.mean(mean_changes):.2f} mg/dL")
    print(f"  Median: {np.median(mean_changes):.2f} mg/dL")
    print(f"  Std: {np.std(mean_changes):.2f} mg/dL")
    print(f"  Range: [{np.min(mean_changes):.2f}, {np.max(mean_changes):.2f}] mg/dL")
    
    print(f"\nSD change statistics:")
    print(f"  Mean: {np.mean(sd_changes):.2f} mg/dL")
    print(f"  Median: {np.median(sd_changes):.2f} mg/dL")
    print(f"  Std: {np.std(sd_changes):.2f} mg/dL")
    print(f"  Range: [{np.min(sd_changes):.2f}, {np.max(sd_changes):.2f}] mg/dL")
    
    # Create plots if requested
    if plot:
        print("\nGenerating plots...")
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # (a) Duration distribution
        best_fit_duration = plot_distribution_with_fits(
            durations, axes[0], 
            '(a) Duration',
            'Duration (minutes)',
            fit_distributions=['expon', 'norm', 'uniform']
        )
        print(f"  Duration best fit: {best_fit_duration}")
        
        # (b) Mean change distribution
        best_fit_mean = plot_distribution_with_fits(
            mean_changes, axes[1],
            '(b) Mean change',
            'Difference in mean (mg/dL)',
            fit_distributions=['norm', 'uniform']
        )
        print(f"  Mean change best fit: {best_fit_mean}")
        
        # (c) SD change distribution
        best_fit_sd = plot_distribution_with_fits(
            sd_changes, axes[2],
            '(c) Standard deviation change',
            'Difference in SD (mg/dL)',
            fit_distributions=['norm', 'uniform']
        )
        print(f"  SD change best fit: {best_fit_sd}")
        
        plt.suptitle('Distribution of Changepoint Properties for OHIO Dataset', 
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFigure saved to: {save_path}")
        
        plt.show()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        'changepoints': changepoints,
        'durations': durations,
        'mean_changes': mean_changes,
        'sd_changes': sd_changes,
        'num_changepoints': len(changepoints) - 2
    }

def load_ohio_xml(xml_path):
    """
    Load glucose data from OHIO XML format
    
    Parameters:
    - xml_path: path to OHIO XML file (e.g., '559-ws-training.xml')
    
    Returns:
    - pandas DataFrame with timestamp and glucose columns
    """
    import xml.etree.ElementTree as ET
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Extract glucose readings
    glucose_data = []
    for event in root.findall('.//glucose_level'):
        timestamp = event.get('ts')
        value = float(event.get('value'))
        glucose_data.append({
            'timestamp': pd.to_datetime(timestamp),
            'glucose': value
        })
    
    df = pd.DataFrame(glucose_data)
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Loaded {len(df)} glucose readings from {xml_path}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df

def load_ohio_csv(csv_path):
    """
    Load glucose data from CSV format
    
    Parameters:
    - csv_path: path to CSV file with 'glucose' column
    
    Returns:
    - numpy array of glucose values
    """
    df = pd.read_csv(csv_path)
    
    if 'glucose' not in df.columns:
        raise ValueError("CSV must have 'glucose' column")
    
    glucose_data = df['glucose'].values
    
    # Remove NaN values
    glucose_data = glucose_data[~np.isnan(glucose_data)]
    
    print(f"Loaded {len(glucose_data)} glucose readings from {csv_path}")
    
    return glucose_data

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("OHIO Glucose Changepoint Distribution Analysis")
    print("="*80)
    
    # --------------------------------------------------------------------
    # OPTION 1: Load from OHIO XML file
    # --------------------------------------------------------------------
    # Uncomment and modify the path to your actual OHIO data file
    """
    xml_path = 'path/to/559-ws-training.xml'
    df = load_ohio_xml(xml_path)
    glucose_data = df['glucose'].values
    """
    
    # --------------------------------------------------------------------
    # OPTION 2: Load from CSV file
    # --------------------------------------------------------------------
    # Uncomment and modify the path to your CSV file
    """
    csv_path = 'path/to/ohio_glucose.csv'
    glucose_data = load_ohio_csv(csv_path)
    """
    
    # --------------------------------------------------------------------
    # OPTION 3: Generate synthetic data for testing
    # --------------------------------------------------------------------
    print("\nGenerating synthetic glucose data for demonstration...")
    np.random.seed(42)
    
    # Simulate ~7 days of glucose data (5-minute intervals)
    n_points = 2016  # 7 days * 24 hours * 12 readings/hour
    
    glucose_data = []
    current_mean = 120  # Starting mean glucose
    current_std = 15    # Starting std
    
    for i in range(n_points):
        # Create changepoints every 100-300 points (8-25 hours)
        if i > 0 and i % np.random.randint(100, 300) == 0:
            # Shift mean (meal effect, insulin, exercise)
            current_mean += np.random.randn() * 25
            current_mean = np.clip(current_mean, 80, 180)  # Keep realistic
            
            # Shift std (variability change)
            current_std = max(5, current_std + np.random.randn() * 5)
        
        # Generate glucose value with some autocorrelation
        if i > 0:
            # Add autocorrelation (glucose doesn't jump randomly)
            value = 0.8 * glucose_data[-1] + 0.2 * np.random.normal(current_mean, current_std)
        else:
            value = np.random.normal(current_mean, current_std)
        
        value = np.clip(value, 40, 400)
        glucose_data.append(value)
    
    glucose_data = np.array(glucose_data)
    print(f"Generated {len(glucose_data)} synthetic glucose readings")
    
    # --------------------------------------------------------------------
    # RUN ANALYSIS
    # --------------------------------------------------------------------
    results = analyze_ohio_changepoints(
        glucose_data,
        plot=True,
        save_path='ohio_changepoint_distributions.png'
    )
    
    # --------------------------------------------------------------------
    # SAVE RESULTS TO CSV
    # --------------------------------------------------------------------
    results_df = pd.DataFrame({
        'duration_minutes': results['durations'],
        'mean_change_mgdl': results['mean_changes'],
        'sd_change_mgdl': results['sd_changes']
    })
    
    results_df.to_csv('changepoint_properties.csv', index=False)
    print("\nChangepoint properties saved to: changepoint_properties.csv")
    
    print("\nDone! Check the generated figure and CSV file.")
