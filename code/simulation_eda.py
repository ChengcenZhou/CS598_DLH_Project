import numpy as np
from sklearn.mixture import GaussianMixture

class NonstationaritySimulator:
    """Implements the nonstationarity simulation methods from the paper"""
    
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.scaler = StandardScaler()
        self.data_scaled = self.scaler.fit_transform(data)
    
    def raw_simulated(self, n_samples):
        """Generate basic simulated data without nonstationarity"""
        # Fit GMM for each class
        gmm_class0 = GaussianMixture(n_components=3).fit(
            self.data_scaled[self.labels == 0])
        gmm_class1 = GaussianMixture(n_components=3).fit(
            self.data_scaled[self.labels == 1])
        
        # Generate samples
        n_per_class = n_samples // 2
        X_sim_0 = gmm_class0.sample(n_per_class)[0]
        X_sim_1 = gmm_class1.sample(n_per_class)[0]
        
        X_sim = np.vstack([X_sim_0, X_sim_1])
        y_sim = np.array([0] * n_per_class + [1] * n_per_class)
        
        return self.scaler.inverse_transform(X_sim), y_sim
    
    def meanshift_constant(self, n_samples, shift_amount=0.5):
        """Apply constant mean shift across time"""
        X_sim, y_sim = self.raw_simulated(n_samples)
        
        # Apply linear shift
        for i in range(X_sim.shape[1]):
            shifts = np.linspace(0, shift_amount, n_samples)
            X_sim[:, i] += shifts * np.std(X_sim[:, i])
        
        return X_sim, y_sim
    
    def meansdshift_constant(self, n_samples, mean_shift=0.5, sd_shift=0.3):
        """Apply constant mean and SD shift"""
        X_sim, y_sim = self.raw_simulated(n_samples)
        
        for i in range(X_sim.shape[1]):
            # Mean shift
            mean_shifts = np.linspace(0, mean_shift, n_samples)
            X_sim[:, i] += mean_shifts * np.std(X_sim[:, i])
            
            # SD shift
            sd_mult = np.linspace(1.0, 1.0 + sd_shift, n_samples)
            X_sim[:, i] = (X_sim[:, i] - np.mean(X_sim[:, i])) * sd_mult + np.mean(X_sim[:, i])
        
        return X_sim, y_sim
    
    def meanshift_varying(self, n_samples, n_changepoints=5):
        """Apply varying mean shifts with changepoints"""
        X_sim, y_sim = self.raw_simulated(n_samples)
        
        changepoints = np.sort(np.random.choice(n_samples, n_changepoints, replace=False))
        changepoints = np.concatenate([[0], changepoints, [n_samples]])
        
        for i in range(X_sim.shape[1]):
            for j in range(len(changepoints) - 1):
                start, end = changepoints[j], changepoints[j+1]
                shift = np.random.uniform(-0.5, 0.5)
                X_sim[start:end, i] += shift * np.std(X_sim[:, i])
        
        return X_sim, y_sim
    
    def meansdshift_varying(self, n_samples, n_changepoints=5):
        """Apply varying mean and SD shifts"""
        X_sim, y_sim = self.raw_simulated(n_samples)
        
        changepoints = np.sort(np.random.choice(n_samples, n_changepoints, replace=False))
        changepoints = np.concatenate([[0], changepoints, [n_samples]])
        
        for i in range(X_sim.shape[1]):
            for j in range(len(changepoints) - 1):
                start, end = changepoints[j], changepoints[j+1]
                # Mean shift
                mean_shift = np.random.uniform(-0.5, 0.5)
                X_sim[start:end, i] += mean_shift * np.std(X_sim[:, i])
                # SD shift
                sd_mult = np.random.uniform(0.7, 1.3)
                seg_mean = np.mean(X_sim[start:end, i])
                X_sim[start:end, i] = (X_sim[start:end, i] - seg_mean) * sd_mult + seg_mean
        
        return X_sim, y_sim
    
    def learned_nonstationarity(self, n_samples, window_size=100):
        """Learn and apply nonstationarity patterns from real data"""
        # Detect changepoints in real data
        changepoints = self._detect_changepoints(window_size)
        
        # Learn distributional shifts
        shifts = self._learn_distributional_shifts(changepoints, window_size)
        
        # Generate base simulation
        X_sim, y_sim = self.raw_simulated(n_samples)
        
        # Apply learned shifts
        X_sim = self._apply_learned_shifts(X_sim, shifts, n_samples)
        
        return X_sim, y_sim
    
    def _detect_changepoints(self, window_size):
        """Detect changepoints using statistical tests"""
        changepoints = [0]
        
        for i in range(window_size, len(self.data_scaled) - window_size, window_size):
            # Compare distributions before and after
            before = self.data_scaled[i-window_size:i]
            after = self.data_scaled[i:i+window_size]
            
            # KS test for distribution change
            _, p_value = stats.ks_2samp(before.flatten(), after.flatten())
            
            if p_value < 0.01:  # Significant change
                changepoints.append(i)
        
        changepoints.append(len(self.data_scaled))
        return changepoints
    
    def _learn_distributional_shifts(self, changepoints, window_size):
        """Learn distribution parameters for each segment"""
        shifts = []
        
        for i in range(len(changepoints) - 1):
            start, end = changepoints[i], changepoints[i+1]
            segment = self.data_scaled[start:end]
            
            shifts.append({
                'mean': np.mean(segment, axis=0),
                'std': np.std(segment, axis=0),
                'start': start,
                'end': end
            })
        
        return shifts
    
    def _apply_learned_shifts(self, X_sim, shifts, n_samples):
        """Apply learned distributional shifts to simulated data"""
        # Scale shifts to match simulated data length
        total_real_length = shifts[-1]['end']
        scale_factor = n_samples / total_real_length
        
        for shift in shifts:
            sim_start = int(shift['start'] * scale_factor)
            sim_end = int(shift['end'] * scale_factor)
            
            if sim_end > n_samples:
                sim_end = n_samples
            
            # Apply shift
            for j in range(X_sim.shape[1]):
                current_mean = np.mean(X_sim[sim_start:sim_end, j])
                current_std = np.std(X_sim[sim_start:sim_end, j])
                
                # Adjust to match learned distribution
                target_mean = shift['mean'][j]
                target_std = shift['std'][j]
                
                X_sim[sim_start:sim_end, j] = (
                    (X_sim[sim_start:sim_end, j] - current_mean) / current_std * target_std + target_mean
                )
        
        return X_sim
