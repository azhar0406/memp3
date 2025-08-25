class SimpleEncoder:
    """Simple text to audio encoder"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Frequency range for encoding (200-2000 Hz)
        self.min_freq = 200
        self.max_freq = 2000
        # Printable ASCII characters (32-126)
        self.chars = ''.join(chr(i) for i in range(32, 127))
        
    def char_to_freq(self, char):
        """Map character to frequency"""
        if char not in self.chars:
            char = '?'  # Default for unknown characters
        idx = self.chars.index(char)
        # Map to frequency range
        freq = self.min_freq + (idx / (len(self.chars) - 1)) * (self.max_freq - self.min_freq)
        return freq
    
    def freq_to_char(self, freq):
        """Map frequency back to character"""
        # Reverse mapping
        idx = round((freq - self.min_freq) / (self.max_freq - self.min_freq) * (len(self.chars) - 1))
        idx = max(0, min(idx, len(self.chars) - 1))  # Clamp to valid range
        return self.chars[idx]
    
    def encode(self, text):
        """Encode text to audio signal"""
        import numpy as np
        
        # Generate audio signal
        duration_per_char = 0.1  # 100ms per character
        total_duration = len(text) * duration_per_char
        t = np.linspace(0, total_duration, int(self.sample_rate * total_duration), endpoint=False)
        
        # Create signal
        signal = np.zeros_like(t)
        segment_length = int(self.sample_rate * duration_per_char)
        
        for i, char in enumerate(text):
            freq = self.char_to_freq(char)
            start_idx = i * segment_length
            end_idx = min(start_idx + segment_length, len(t))
            if start_idx < len(t):
                segment_t = t[start_idx:end_idx]
                signal[start_idx:end_idx] = np.sin(2 * np.pi * freq * segment_t)
        
        # Normalize
        signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
        return signal
    
    def decode(self, signal):
        """Decode audio signal to text"""
        import numpy as np
        from scipy.signal import find_peaks
        
        # Extract characters
        duration_per_char = 0.1
        segment_length = int(self.sample_rate * duration_per_char)
        num_chars = len(signal) // segment_length
        
        text = ""
        for i in range(num_chars):
            start_idx = i * segment_length
            end_idx = min(start_idx + segment_length, len(signal))
            segment = signal[start_idx:end_idx]
            
            # Skip empty segments
            if len(segment) == 0:
                continue
                
            # Find dominant frequency using FFT
            fft = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
            magnitudes = np.abs(fft)
            
            # Find the frequency with maximum magnitude
            if len(magnitudes) > 0:
                max_idx = np.argmax(magnitudes)
                peak_freq = freqs[max_idx]
                char = self.freq_to_char(peak_freq)
                text += char
        
        return text