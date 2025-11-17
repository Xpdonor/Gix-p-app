import os
import numpy as np
import wave
import struct

try:
    import miniaudio
    MINIAUDIO_AVAILABLE = True
except ImportError:
    MINIAUDIO_AVAILABLE = False
    print("miniaudio not available - MP3 support disabled")

def read_audio_file(path):
    
    try:
        
        if path.lower().endswith('.mp3') and MINIAUDIO_AVAILABLE:
            # استفاده از miniaudio برای decode کردن MP3
            audio_data = miniaudio.decode_file(path)
            data = np.frombuffer(audio_data.samples, dtype=np.float32)
            
            if audio_data.nchannels > 1:
                data = data.reshape(-1, audio_data.nchannels)
                data = data.mean(axis=1)
            
            return data, audio_data.sample_rate
        
        
        elif path.lower().endswith('.wav'):
            with wave.open(path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_width = wav_file.getsampwidth()
                framerate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                
                raw_data = wav_file.readframes(frames)
                
                if sample_width == 2:  # 16-bit
                    data = np.frombuffer(raw_data, dtype=np.int16)
                elif sample_width == 4:  # 32-bit
                    data = np.frombuffer(raw_data, dtype=np.int32)
                else:  # 8-bit
                    data = np.frombuffer(raw_data, dtype=np.uint8)
                
                data = data.astype(np.float32)
                if sample_width == 2:
                    data /= 32768.0
                elif sample_width == 4:
                    data /= 2147483648.0
                else:
                    data = (data - 128) / 128.0
                
                if channels > 1:
                    data = data.reshape(-1, channels)
                    data = data.mean(axis=1)
                
                return data, framerate
        
        else:
           
            if MINIAUDIO_AVAILABLE:
                audio_data = miniaudio.decode_file(path)
                data = np.frombuffer(audio_data.samples, dtype=np.float32)
                
                if audio_data.nchannels > 1:
                    data = data.reshape(-1, audio_data.nchannels)
                    data = data.mean(axis=1)
                
                return data, audio_data.sample_rate
            else:
                raise NotImplementedError(f"Unsupported audio format: {path}")
            
    except Exception as e:
        print(f"Error reading audio file {path}: {e}")
        
        return np.zeros(44100), 44100

def write_audio_file(path, data, sr):
    
    try:
        data = data * 32767
        data = np.clip(data, -32768, 32767)
        data = data.astype(np.int16)
        
        with wave.open(path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sr)
            wav_file.writeframes(data.tobytes())
            
    except Exception as e:
        print(f"Error writing audio file: {e}")

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
        y, orig_sr = read_audio_file(src_path)
        
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
        
        write_audio_file(dst_path, out, sr)
        return dst_path
        
    except Exception as e:
        print("apply_eq failed", e)
        raise

def spectral_centroid(path, sr=22050):
    try:
        y, file_sr = read_audio_file(path)
        
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

