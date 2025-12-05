class GlucoseNonstationaritySimulator:
    """Nonstationarity simulator for glucose data"""
    
    def __init__(self, data):
        self.data = data
        self.mean = np.mean(data)
        self.std = np.std(data)
    
    def raw_simulated(self, n_samples):
        """Generate basic AR(1) process"""
        # Fit AR(1) model
        phi = np.corrcoef(self.data[:-1], self.data[1:])[0, 1]
        
        # Generate samples
        X_sim = np.zeros(n_samples)
        X_sim[0] = self.mean
        
        for i in range(1, n_samples):
            X_sim[i] = self.mean + phi * (X_sim[i-1] - self.mean) + np.random.normal(0, self.std)
        
        return X_sim
    
    def meanshift_constant(self, n_samples):
        """Apply constant mean shift"""
        X_sim = self.raw_simulated(n_samples)
        
        # Linear shift in mean
        shifts = np.linspace(0, 20, n_samples)  # Up to 20 mg/dL shift
        X_sim += shifts
        
        return X_sim
    
    def meansdshift_constant(self, n_samples):
        """Apply constant mean and SD shift"""
        X_sim = self.raw_simulated(n_samples)
        
        # Mean shift
        mean_shifts = np.linspace(0, 20, n_samples)
        X_sim += mean_shifts
        
        # SD shift
        sd_mult = np.linspace(1.0, 1.5, n_samples)
        X_sim = (X_sim - np.mean(X_sim)) * sd_mult + np.mean(X_sim)
        
        return X_sim
    
    def learned_nonstationarity(self, n_samples, window_size=288):
        """Learn meal and activity patterns from real data"""
        # Detect meal events (rapid glucose increases)
        meal_times = self._detect_meal_events()
        
        # Detect activity patterns (glucose variability changes)
        activity_patterns = self._detect_activity_patterns(window_size)
        
        # Generate base simulation
        X_sim = self.raw_simulated(n_samples)
        
        # Apply meal effects
        X_sim = self._apply_meal_effects(X_sim, meal_times, n_samples)
        
        # Apply activity patterns
        X_sim = self._apply_activity_patterns(X_sim, activity_patterns, n_samples)
        
        return X_sim
    
    def _detect_meal_events(self):
        """Detect meal events from glucose spikes"""
        diff = np.diff(self.data)
        meal_threshold = np.percentile(diff, 95)
        meal_times = np.where(diff > meal_threshold)[0]
        return meal_times
    
    def _detect_activity_patterns(self, window_size):
        """Detect activity patterns from glucose variability"""
        patterns = []
        
        for i in range(0, len(self.data) - window_size, window_size):
            window = self.data[i:i+window_size]
            pattern = {
                'mean': np.mean(window),
                'std': np.std(window),
                'trend': np.polyfit(range(len(window)), window, 1)[0]
            }
            patterns.append(pattern)
        
        return patterns
    
    def _apply_meal_effects(self, X_sim, meal_times, n_samples):
        """Apply meal-like glucose spikes"""
        scale_factor = n_samples / len(self.data)
        
        for meal_time in meal_times:
            sim_time = int(meal_time * scale_factor)
            if sim_time < n_samples - 36:  # 3-hour effect
                # Spike pattern: rapid rise, gradual fall
                spike = np.concatenate([
                    np.linspace(0, 40, 6),  # 30-min rise
                    np.linspace(40, 0, 30)  # 2.5-hour fall
                ])
                end_time = min(sim_time + len(spike), n_samples)
                X_sim[sim_time:end_time] += spike[:end_time-sim_time]
        
        return X_sim
    
    def _apply_activity_patterns(self, X_sim, patterns, n_samples):
        """Apply learned activity patterns"""
        segment_length = n_samples // len(patterns)
        
        for i, pattern in enumerate(patterns):
            start = i * segment_length
            end = min(start + segment_length, n_samples)
            
            # Adjust mean and variance
            current_mean = np.mean(X_sim[start:end])
            current_std = np.std(X_sim[start:end])
            
            X_sim[start:end] = (
                (X_sim[start:end] - current_mean) / current_std * pattern['std'] + pattern['mean']
            )
            
            # Add trend
            X_sim[start:end] += pattern['trend'] * np.arange(end - start)
        
        return X_sim
