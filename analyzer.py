import os
import numpy as np
import soundfile as sf

def butter_bandpass(lowcut, highcut, fs, order=4):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
   
    n = order * 8  
    t = np.arange(n) - n//2
    t = t + 0.5 if n % 2 == 0 else t
    
    
    h = np.sinc(2 * high * t) - np.sinc(2 * low * t)
    
    
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    h = h * window
    
    
    h = h / np.sum(h)
    
    return h, [1.0]  

def lfilter(b, a, x):
    
    
    if len(a) == 1 and a[0] == 1.0:
        y = np.convolve(x, b, mode='same')
        return y
    else:
        
        y = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(min(len(b), i+1)):
                y[i] += b[j] * x[i-j]
            for j in range(1, min(len(a), i+1)):
                y[i] -= a[j] * y[i-j]
            if len(a) > 0:
                y[i] /= a[0]
        return y

def simple_resample(y, orig_sr, target_sr):
    
    if orig_sr == target_sr:
        return y
    
    
    ratio = target_sr / orig_sr
    n_samples = int(len(y) * ratio)
    
    
    x_old = np.arange(len(y))
    x_new = np.linspace(0, len(y)-1, n_samples)
    
    
    y_resampled = np.interp(x_new, x_old, y)
    
    return y_resampled

def apply_eq_and_write(src_path, dst_path, gains_db, sr=22050):
    """
    gains_db: {'low': db, 'mid': db, 'high': db}
    Applies simple 3-band EQ and writes dst_path (wav).
    """
    try:
        y, orig_sr = sf.read(src_path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        
        
        if orig_sr != sr:
            y = simple_resample(y, orig_sr, sr)

        bands = {'low': (20, 250), 'mid': (250, 4000), 'high': (4000, sr//2 - 100)}
        out = np.zeros_like(y)
        
        for band, (lo, hi) in bands.items():
            b, a = butter_bandpass(max(20, lo), min(hi, sr//2-10), sr, order=4)
            band_sig = lfilter(b, a, y)
            gain = 10 ** (gains_db.get(band, 0.0) / 20.0)
            out += band_sig * gain
        
        
        maxv = np.max(np.abs(out)) + 1e-9
        out = out / maxv * 0.95
        sf.write(dst_path, out, sr)
        return dst_path
    except Exception as e:
        print("apply_eq failed", e)
        raise

def spectral_centroid(path, sr=22050):
    """محاسبه centroid طیفی بدون librosa"""
    try:
        y, file_sr = sf.read(path)
        if y.ndim > 1:
            y = y.mean(axis=1)
        
        
        if file_sr != sr:
            y = simple_resample(y, file_sr, sr)
        
        
        if len(y) > 30 * sr:
            y = y[:30 * sr]
        
        
        frame_size = 2048
        hop_size = 512
        n_frames = (len(y) - frame_size) // hop_size + 1
        
        spectrogram = []
        for i in range(n_frames):
            start = i * hop_size
            end = start + frame_size
            frame = y[start:end] * np.hanning(frame_size)
            spectrum = np.abs(np.fft.rfft(frame))
            spectrogram.append(spectrum)
        
        spectrogram = np.array(spectrogram)
        freqs = np.fft.rfftfreq(frame_size, 1/sr)
        
        
        sc_values = []
        for spectrum in spectrogram:
            if np.sum(spectrum) > 1e-9:
                centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
                sc_values.append(centroid)
        
        return float(np.mean(sc_values)) if sc_values else 2500.0
        
    except Exception as e:
        print("sc fail", e)
        return 2500.0
