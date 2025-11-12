
import os
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter
from librosa.core import resample as lr_resample
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_eq_and_write(src_path, dst_path, gains_db, sr=22050):
    """
    gains_db: {'low': db, 'mid': db, 'high': db}
    Applies simple 3-band EQ and writes dst_path (wav).
    """
    try:
        y, orig_sr = sf.read(src_path, dtype='float32')
        if y.ndim > 1:
            y = y.mean(axis=1)
        # resample if needed (naive)
        if orig_sr != sr:
            import librosa
            y = lr_resample(y, orig_sr, sr, res_type='kaiser_best', fix=True, scale=False)


        bands = {'low': (20, 250), 'mid': (250, 4000), 'high': (4000, sr//2 - 100)}
        out = np.zeros_like(y)
        for band, (lo, hi) in bands.items():
            b, a = butter_bandpass(max(20, lo), min(hi, sr//2-10), sr, order=4)
            band_sig = lfilter(b, a, y)
            gain = 10 ** (gains_db.get(band, 0.0) / 20.0)
            out += band_sig * gain
        # normalize
        maxv = np.max(np.abs(out)) + 1e-9
        out = out / maxv * 0.95
        sf.write(dst_path, out, sr)
        return dst_path
    except Exception as e:
        print("apply_eq failed", e)
        raise

def spectral_centroid(path, sr=22050):
    try:
        import librosa
        y, _ = librosa.load(path, sr=sr, mono=True, duration=30)
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)
        return float(np.mean(sc))
    except Exception as e:
        print("sc fail", e)
        return 2500.0
